import math
from collections import namedtuple
from queue import Queue
import cv2
import numpy as np
from rl_trainer.algo.ppo import PPO
from rl_trainer.log_path import make_logpath

#parameters
init_angle = 30.
zero_angle = -180.
view_size = 12.5
scaled_size = 50
scale_level = 2
map_size = 2000
direction_dim = 19
pos_dim = 3
target_num = 3
history_num = 3
target_piece = 10
#8 directions of a pixel
dx = [-1, 0, 1, 0, -1, -1, 1, 1]
dy = [0, -1, 0, 1, -1, 1, -1, 1]

#some auxiliary functions
#radius of agent on different maps
def radius_on_maps(radius):
    radius_list = [9.375, 12.5, 15, 18.75, 20]
    return sorted(radius_list, key=lambda x: (x - radius * 6) ** 2)[0] / 6

def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def direction(p1, p2):
    if distance(p1, p2) < 1e-10: return 0.
    angle = math.acos((p2[0] - p1[0]) / distance(p1, p2)) * 180 / math.pi
    return 180 - angle if p2[1] > p1[1] else -(180 - angle)

#compare similarity of 2 imgs and return the offset
def get_offset(img1, img2, position):  
    diff = 1e10 
    offset = None
    r = 9
    #search with x or y offset in [-r/2,r/2] 
    for i in range(r):
        for j in range(r):
            nx = int(i + position[0] - r // 2)
            ny = int(j + position[1] - r // 2)
            if nx > 0:
                if ny > 0:
                    image1 = img1[:-nx, :-ny]
                    image2 = img2[nx:, ny:]
                elif ny < 0:
                    image1 = img1[:-nx, -ny:]
                    image2 = img2[nx:, :ny]
                else:
                    image1 = img1[:-nx, :]
                    image2 = img2[nx:, :]
            elif nx < 0:
                if ny > 0:
                    image1 = img1[-nx:, :-ny]
                    image2 = img2[:nx, ny:]
                elif ny < 0:
                    image1 = img1[-nx:, -ny:]
                    image2 = img2[:nx, :ny]
                else:
                    image1 = img1[-nx:, :]
                    image2 = img2[:nx, :]
            else:
                if ny > 0:
                    image1 = img1[:, :-ny]
                    image2 = img2[:, ny:]
                elif ny < 0:
                    image1 = img1[:, -ny:]
                    image2 = img2[:, :ny]
                else:
                    image1 = img1
                    image2 = img2
            diff_sum = np.sum((image1 - image2) ** 2, where=np.logical_and(np.logical_and(image1 >= 0, image1 < 150),
                                                                           np.logical_and(image2 >= 0, image2 < 150)))
            if diff > diff_sum:
                offset = (nx, ny)
                diff = diff_sum
    return offset 

class RuleAgent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.radius = None
        self.past = None
        self.game_map = dict()
        self.game_map["objects"] = list()
        self.game_map["agents"] = list()
        self.game_map["view"] = {
            "width": 600,
            "height": 600,
            "edge": 50
        }
        self.last_action = None
        self.v = 0.    
        self.x = 0.
        self.y = 0.
        self.angle = zero_angle
        self.end_flag = False
        self.arrows = []
        self.points = None
        self.force = 0
        self.history = []
        self.target_history = []
        self.global_map = np.ones([map_size, map_size, 4], dtype=np.float32) * 150.
        self.run_dir, _ = make_logpath("olympics-running", "ppo")
        self.episode = 0
        self.model = PPO(self.run_dir)
        self.transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    def reset(self):
        self.radius = None
        self.past = None
        self.game_map = dict()
        self.game_map["objects"] = list()
        self.game_map["agents"] = list()
        self.game_map["view"] = {
            "width": 600,
            "height": 600,
            "edge": 50
        }
        self.last_action = None
        self.v = 0.
        self.x = 0.
        self.y = 0.
        self.angle = zero_angle
        self.force = 0
        self.end_flag = False
        self.points = None
        self.arrows = []
        self.history = []
        self.target_history = []
        self.global_map = np.ones([map_size, map_size, 4], dtype=np.float32) * 150.
        self.done = True
        self.obs_ctrl_agent = np.zeros([22])

    def process_obs(self, obs):  
        processed_obs = np.zeros(obs.shape + (4,), dtype=np.float32) 
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i, j] == 1 or obs[i, j] == 5:
                    pass  
                elif obs[i, j] == 4:
                    processed_obs[i, j, 0] = 1.
                    processed_obs[i, j, 3] = 60   
                elif obs[i, j] == 6:
                    processed_obs[i, j, 1] = 1.
                    processed_obs[i, j, 3] = 120  
                elif obs[i, j] == 7:
                    processed_obs[i, j, 2] = 1.
                    processed_obs[i, j, 3] = 90   
                    self.end_flag = True
        return processed_obs

    def rotate(self, img, angle):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        obs_warped = np.zeros(img.shape, dtype=np.float32)
        for i in range(img.shape[2]):
            obs_warped[:, :, i] = cv2.warpAffine(img[:, :, i], M, (cols, rows), borderValue=-0.00001)
        return obs_warped

    def update_map(self, img, pos_x, pos_y):
        x = int(pos_x + 0.5)
        y = int(pos_y + 0.5)
        for i in range(scaled_size):
            for j in range(scaled_size):
                if img[i, j, 3] >= 0 and self.global_map[
                    x + i - scaled_size // 2 + map_size // 2, y + j - scaled_size // 2 + map_size // 2, 3] >= 150:
                    self.global_map[
                        x + i - scaled_size // 2 + map_size // 2, y + j - scaled_size // 2 + map_size // 2] = img[i, j]

    def update_self_arround(self, r_scale=1.2):
        r = self.radius * scale_level * r_scale
        for i in range(int(r * 2 + 1)):
            for j in range(int(r * 2 + 1)):
                x = int(self.x + i - r) + map_size // 2
                y = int(self.y + j - r) + map_size // 2
                if distance((int(self.x + i - r), int(self.y + j - r)), (self.x, self.y)) <= r and self.global_map[
                    x, y, 3] >= 150:
                    self.global_map[x, y] = 0

    def add_arrows(self, img, now_x, now_y):
        arrow_map = np.zeros_like(img, dtype=int)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > 0.9 and arrow_map[i, j] == 0:
                    x1, x2, y1, y2 = i, i + 1, j, j + 1
                    q = Queue()
                    q.put((i, j))
                    while not q.empty():
                        x, y = q.get()
                        if x1 > x:
                            x1 = x
                        if x2 < x + 1:
                            x2 = x + 1
                        if y1 > y:
                            y1 = y
                        if y2 < y + 1:
                            y2 = y + 1
                        for k in range(4):
                            nx = dx[k] + x
                            ny = dy[k] + y
                            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and img[nx, ny] > 0.9 and arrow_map[
                                nx, ny] == 0:
                                arrow_map[nx, ny] = 1
                                q.put((nx, ny))
                    arrow_direction = None
                    e_list, w_list, n_list, s_list = [], [], [], []
                    e = 0
                    for k in range(x1, x2):
                        my1 = y2
                        my2 = y1 - 1
                        b = 0
                        c = 0
                        d = 0
                        for ll in range(y1, y2):
                            if arrow_map[k, ll] == 1:
                                b = 1
                                if my1 > ll:
                                    my1 = ll
                                if my2 < ll:
                                    my2 = ll
                            if arrow_map[k, ll] == 0 and b > 0:
                                c = 1
                            if arrow_map[k, ll] == 1 and c > 0:
                                d = 1
                        e_list.append(my1)
                        w_list.append(my2)
                        if d > 0:
                            e = 1
                    if e == 0:
                        if e_list[0] < max(e_list) and e_list[-1] < max(e_list):
                            arrow_direction = 'E'
                            pos = (e_list.index(max(e_list)) + x1 - img.shape[0] // 2 + now_x,
                                   max(e_list) - img.shape[1] // 2 + now_y)
                        if w_list[0] > min(w_list) and w_list[-1] > min(w_list):
                            arrow_direction = 'W'
                            pos = (w_list.index(min(w_list)) + x1 - img.shape[0] // 2 + now_x,
                                   min(w_list) - img.shape[1] // 2 + now_y)
                    e = 0
                    for k in range(y1, y2):
                        mx1 = x2
                        mx2 = x1 - 1
                        b = 0
                        c = 0
                        d = 0
                        for ll in range(x1, x2):
                            if arrow_map[ll, k] == 1:
                                b = 1
                                if mx1 > ll:
                                    mx1 = ll
                                if mx2 < ll:
                                    mx2 = ll
                            if arrow_map[ll, k] == 0 and b > 0:
                                c = 1
                            if arrow_map[ll, k] == 1 and c > 0:
                                d = 1
                        s_list.append(mx1)
                        n_list.append(mx2)
                        if d > 0:
                            e = 1
                    if e == 0:
                        if s_list[0] < max(s_list) and s_list[-1] < max(s_list):
                            arrow_direction = 'S'
                            pos = (max(s_list) - img.shape[0] // 2 + now_x,
                                   s_list.index(max(s_list)) + y1 - img.shape[1] // 2 + now_y)
                        if n_list[0] > min(n_list) and n_list[-1] > min(n_list):
                            arrow_direction = 'N'
                            pos = (min(n_list) - img.shape[0] // 2 + now_x,
                                   n_list.index(min(n_list)) + y1 - img.shape[1] // 2 + now_y)
                    if arrow_direction is not None:
                        redundant = False
                        for arrow in self.arrows:
                            if distance(arrow[0], pos) < 10:
                                redundant = True
                        if not redundant:
                            self.arrows.append([pos, arrow_direction, True])
                    for k in range(len(self.arrows)):
                        for ll in range(len(self.arrows)):
                            if k != ll and distance(self.arrows[k][0], self.arrows[ll][0]) < 60 and self.in_direction(
                                    self.arrows[ll], self.arrows[k][0]) > 0:
                                self.arrows[ll][2] = False

    def curr_pos(self):
        return int(self.x) + map_size // 2, int(self.y) + map_size // 2
        
    def get_angle_offset(self, point):
        angle = direction(self.curr_pos(), point)
        angle_offset = angle - self.angle
        if angle_offset > 180: angle_offset -= 360
        if angle_offset < -180: angle_offset += 360
        return -abs(angle_offset) / 180 * 0.5

    def get_distance_score(self, point):
        d = distance(self.curr_pos(), point)
        return -d / 50 if d > 70 else 0.

    def get_edges(self):
        img = self.global_map
        edge_map = np.zeros(img.shape[:2], dtype=int)
        past_x = np.ones(img.shape[:2], dtype=int) * -1
        past_y = np.ones(img.shape[:2], dtype=int) * -1
        x = self.x
        y = self.y
        start_x, start_y = int(x) + map_size // 2, int(y) + map_size // 2
        q = Queue()
        q.put((start_x, start_y))
        edge_map[start_x, start_y] = 1
        edge_point_list = []
        while not q.empty():
            x, y = q.get()
            if img[x, y, 2] > 0.1:
                tx, ty = x, y
                point_list = [(x, y)]
                while tx != start_x or ty != start_y:
                    tx, ty = past_x[tx, ty], past_y[tx, ty]
                    point_list.append((tx, ty))
                return point_list
            for k in range(4):
                nx = dx[k] * 3 + x
                ny = dy[k] * 3 + y
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and edge_map[nx, ny] == 0:
                    if img[nx, ny, 1] <= 0:
                        edge_map[nx, ny] = 1
                        q.put((nx, ny))
                        past_x[nx, ny] = x
                        past_y[nx, ny] = y
                    elif img[nx, ny, 1] >= 150:
                        edge_map[nx, ny] = 2
                        edge_point_list.append((nx, ny))
                        past_x[nx, ny] = x
                        past_y[nx, ny] = y
        edge_list = []
        for ep in edge_point_list:
            if edge_map[ep[0], ep[1]] == 2:
                center_x, center_y = 0, 0
                edge = {'points': []}
                x1, y1 = ep[0], ep[1]
                q = Queue()
                q.put((x1, y1))
                edge_map[x1, y1] = 3
                while not q.empty():
                    x, y = q.get()
                    edge['points'].append((x, y))
                    center_x += x
                    center_y += y
                    for k in range(8):
                        nx = dx[k] * 3 + x
                        ny = dy[k] * 3 + y
                        if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and edge_map[nx, ny] == 2 and len(
                                edge['points']) < 20:
                            edge_map[nx, ny] = 3
                            q.put((nx, ny))
                if len(edge['points']) > 4:
                    edge['center'] = (center_x / len(edge['points']), center_y / len(edge['points']))
                    edge_list.append(edge)
        if len(edge_list) == 0: return None
        reward = -100000
        e = None
        for edge in edge_list:
            num = 0
            dist = 1e10
            for arrow in self.arrows:
                d = distance(edge['center'], arrow[0])
                flag = self.in_direction(arrow, edge['center'])
                if arrow[2] and d < 70 and flag > 0:
                    num += 1
                    if dist > d + 1: dist = d + 1
            num += self.get_angle_offset(edge['center'])
            num += self.get_distance_score(edge['center'])
            v = num if num < 0 else num / dist
            if reward < v:
                reward = v
                e = edge
        point = None
        dis = 1e10
        for p in e['points']:
            d = distance(p, e['center'])
            if dis > d:
                dis = d
                point = p
        tx, ty = point[0], point[1]
        point_list = [(point[0], point[1])]
        while tx != start_x or ty != start_y:
            tx, ty = past_x[tx, ty], past_y[tx, ty]
            point_list.append((tx, ty))
        return point_list

    def in_direction(self, arrow, point):
        if distance(arrow[0], point) < 10: return 1
        dis_x = point[0] - arrow[0][0]
        dis_y = point[1] - arrow[0][1]
        if arrow[1] == 'E':
            if dis_y > 0 and abs(dis_x) < abs(dis_y) * 5: return 1
            if dis_y < 0 and abs(dis_x) < abs(dis_y) * 5: return -10
        elif arrow[1] == 'S':
            if dis_x > 0 and abs(dis_y) < abs(dis_x) * 5: return 1
            if dis_x < 0 and abs(dis_y) < abs(dis_x) * 5: return -10
        elif arrow[1] == 'W':
            if dis_y < 0 and abs(dis_x) < abs(dis_y) * 5: return 1
            if dis_y > 0 and abs(dis_x) < abs(dis_y) * 5: return -10
        elif arrow[1] == 'N':
            if dis_x < 0 and abs(dis_y) < abs(dis_x) * 5: return 1
            if dis_x > 0 and abs(dis_y) < abs(dis_x) * 5: return -10
        return 0

    def fix_action(self, force, angle):
        def min_angle(angle1, angle2):
            offset = angle1 - angle2
            if offset > 180: offset -= 360
            elif offset < -180: offset += 360
            return offset

        if len(self.history) > 3:
            a = min_angle(direction(self.history[-4], self.curr_pos()), self.angle)
            actions_map = [0, 10, 20, 30, 40]
            length = len(self.points)
            next_obs_ctrl_agent = np.array([a/180, self.v/5]+[distance(self.points[int(i/10*length)], self.curr_pos())/((i+1)/10*length*10) for i in range(10)]+[min_angle(direction(self.curr_pos(), self.points[int(i/10*length)]), self.angle)/180 for i in range(10)])
            action_ctrl_raw, action_prob= self.model.select_action(self.obs_ctrl_agent, True)
            action = actions_map[action_ctrl_raw]
            post_reward = -1
            trans = self.transition(self.obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward,
                               next_obs_ctrl_agent, self.done)
            self.model.store_transition(trans)
            self.obs_ctrl_agent = next_obs_ctrl_agent
            if self.done:
                if len(self.model.buffer) >= self.model.batch_size: self.model.update(0)
                if self.episode % 10 == 0: self.model.save(self.run_dir, self.episode)
                self.episode += 1
                self.done = False
            if abs(angle) < 29 and abs(a) > action and self.v > 2.5:
                new_angle = angle - a * min(abs(a), 20) / 8
                if new_angle > 30: new_angle = 30
                elif new_angle < -30: new_angle = -30
                angle = new_angle
        return force, angle

    def get_angle_from_curr(self, point):
        angle = direction(self.curr_pos(), point) - self.angle
        if angle > 180: angle -= 360
        elif angle < -180: angle += 360
        if angle > 30: angle = 30
        elif angle < -30: angle = -30
        return angle

    def get_action(self, points):
        if self.v <= 2.01: force = 200
        else: force = 200 / self.v * 2
        pos = self.curr_pos()
        for p in points[:-1]:
            d = int(distance(p, pos))
            flag = True
            for i in range(d):
                x = int((p[0] - pos[0]) / d * i + pos[0])
                y = int((p[1] - pos[1]) / d * i + pos[1])
                if self.global_map[x, y, 1] > 0:
                    flag = False
            if flag: return force, self.get_angle_from_curr(p)
        return force, self.get_angle_from_curr(p)

    def act(self, obs):
        obs = self.process_obs(obs)
        n_obs = cv2.resize(obs[:, :], (scaled_size, scaled_size))
        if self.radius is None:
            force = 0
            angle = init_angle
            obs_warped = self.rotate(n_obs, zero_angle)
            self.radius = 2.5
            self.update_map(obs_warped, (self.radius + view_size) * scale_level, 0)
            self.update_self_arround(1.5)
            nx, ny = (self.radius + view_size) * scale_level, 0
            self.add_arrows(obs_warped[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
            points = self.get_edges()
            if points is None or len(points) < 2: force, angle = self.force, 0
            else:
                force, angle = self.get_action(points)
                self.points = points
        else:
            obs_warped = self.rotate(n_obs, -self.angle)
            if self.radius < 0:
                pos = get_offset(obs_warped[:, :, 3], self.past[:, :, 3], np.array([-4, -14], dtype=np.float32))
                dis = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
                self.radius = radius_on_maps(dis / 0.5 / 2 * math.sin(math.pi * 5 / 12) - view_size)
                self.update_map(self.past, (self.radius + view_size) * scale_level, 0)
                nx, ny = (self.radius + view_size) * scale_level, 0
                self.add_arrows(self.past[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
                nx, ny = (self.radius + view_size) * scale_level + pos[0], pos[1]
                self.update_map(obs_warped, nx, ny)
            else:
                x = self.x - math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                y = self.y - math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                pos = get_offset(obs_warped[:, :, 3],
                                   self.global_map[int(x) - scaled_size // 2 + map_size // 2:int(
                                       x) + scaled_size // 2 + map_size // 2,
                                   int(y) - scaled_size // 2 + map_size // 2:int(
                                       y) + scaled_size // 2 + map_size // 2, 3],
                                   np.array([0, 0], dtype=np.float32))
                nx, ny = x + pos[0], y + pos[1]
                self.x = nx + math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                self.y = ny + math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                self.update_map(obs_warped, nx, ny)
            self.update_self_arround()
            self.add_arrows(obs_warped[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
            points = self.get_edges()
            if points is None or len(points) < 2: force, angle = self.force, 0
            else:
                force, angle = self.get_action(points)
                self.points = points
        self.past = obs_warped
        self.history.append(self.curr_pos())
        self.target_history.append(np.array(self.points))
        if len(self.history) > 3: self.v = distance(self.history[-4], self.history[-1]) / 3
        else: self.v = 1
        force, angle = self.fix_action(force, angle)
        self.last_action = [force, angle]
        self.force = force
        self.angle += angle
        if self.angle < -180: self.angle += 360.
        if self.angle > 180: self.angle -= 360.
        return [force, angle]


agent = RuleAgent()
agent.reset()