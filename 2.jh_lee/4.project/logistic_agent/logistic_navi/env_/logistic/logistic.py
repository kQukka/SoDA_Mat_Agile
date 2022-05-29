import copy
import pandas as pd
from string import ascii_uppercase
from .common import *


class LogisticEnv:
    def __init__(self, height, width, path_item, path_obstacle):
        '''
        height : 그리드 높이
        width : 그리드 너비
        '''
        self.NAME_ITEM = list(ascii_uppercase)[:17]
        self.REWARD = REWARD()

        self.height = height
        self.width = width
        self.num_action = len(IDX_ACTION)

        self.__grid = [[ID_GRID_FLOOR for _ in range(self.width)] for _ in range(self.height)]
        self.__p_item = {}
        self.__p_obstacle = []
        self.__p_order = {}

        self.__p_start = None
        self.__p_goal = None
        self.__p_current = None

        self.__sum_reward = 0
        self.__action = []

        self.__set_p_item(path_item)
        self.__set_p_obstacle(path_obstacle)
        self.__init_grid()

    # region PUBLIC
    '''set start, end point'''
    def set_route(self, p_start, p_goal):
        self.__set_grid_start(p_start)
        self.__set_grid_goal(p_goal)
        self.__p_start = p_start
        self.__p_goal = p_goal
        return True

    '''set order point'''
    def set_p_order(self, name_order: list):
        self.__p_order.clear()
        if not len(name_order):
            return False
        for name in name_order:
            p_item = self.__p_item.get(name)
            # order item point
            self.__p_order[name] = p_item
        return True

    '''start point, grid, action 초기화'''
    def reset(self):
        if (self.__p_start is None) or (self.__p_goal is None):
            return False

        self.__init_grid()
        self.__sum_reward = 0
        self.__p_current = [self.__p_start[0], self.__p_start[1]]
        self.__action.clear()
        self.__action.append((self.__p_start[0], self.__p_start[1]))
        return self.__p_start[0], self.__p_start[1]

    '''action에 따라 step 진행'''
    def step(self, action):
        """
        action에 따라 step을 진행한다.
        :param action: 에이전트 행동
        :return:
            (new_y, new_x), new state
            reward, 리워드
            done, 종료 여부
            result_step, step 결과
        :rtype: numpy.ndarray, float, float, bool, int
        """
        if not self.__p_current:
            return False, False, False, False, False
        cur_y, cur_x = self.__p_current

        new_y, new_x = self.__apply_action(action, cur_y, cur_x)
        result_step, reward, done = self.__get_reward(cur_y, cur_x, new_y, new_x)
        self.__action.append((new_y, new_x))

        self.__set_grid_reward(result_step, cur_y, cur_x, new_y, new_x)
        self.__sum_reward += reward

        if result_step == ID_OUT_GRID:
            new_y = cur_y
            new_x = cur_x
        if (result_step == ID_GENERAL_MOVE) or (result_step == ID_RETURN):
            self.__p_current = (new_y, new_x)
        return (new_y, new_x), reward, done, result_step

    def get_gird(self):
        return copy.deepcopy(self.__grid)

    def get_action(self):
        return copy.deepcopy(self.__action)

    '''get point by item name'''
    def get_p_item(self, name_item):
        if name_item not in self.__p_item:
            return False
        return self.__p_item[name_item]

    def get_result(self):
        data_save=[]
        data_save.append([])
        data_save.append(['REWARD'])
        reward = [self.REWARD.NONE, self.REWARD.NOT_MOVE, self.REWARD.MOVE, self.REWARD.MOVE_SUB,
                  self.REWARD.OBSTACLE, self.REWARD.OUT_GIRD, self.REWARD.RETURN, self.REWARD.GOAL]
        for idx, name in enumerate(self.REWARD.NAME):
            data_save.append([f'{name} : {reward[idx]}'])

        data_save.append([])
        data_save.append(['Grid'])
        for row in self.__grid:
            data_save.append(row)

        data_save.append([])
        data_save.append(['point'])
        data_save.append([f'start : {self.__p_start}, goal : {self.__p_goal}, current : {self.__p_current}'])

        data_save.append([])
        data_save.append(['action'])
        data_save.append(self.__action)
        return data_save

    # endregion PUBLIC

    # region PRIVATE
    # region SET_POINT
    '''item point 정보 추출, grid에 item 표시'''
    def __set_p_item(self, path_csv):
        self.__p_item.clear()
        box_data = pd.read_csv(path_csv)
        for info_box in box_data.itertuples(index=True):
            self.__p_item[info_box.item] = (info_box.row, info_box.col)

    '''grid에 obstacle 표시'''
    def __set_p_obstacle(self, path_csv):
        self.__p_obstacle.clear()
        obstacles_data = pd.read_csv(path_csv)
        for info_obs in obstacles_data.itertuples(index=True):
            self.__p_obstacle.append((info_obs.row, info_obs.col))
    # endregion SET_POINT

    # region GRID
    '''grid init'''
    def __init_grid(self):
        self.__grid = [[ID_GRID_FLOOR for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.__p_item.values():
            self.__grid[x][y] = ID_GRID_ITEM_EMPTY
        if self.__p_order:
            for x, y in self.__p_order.values():
                self.__grid[x][y] = ID_GRID_ITEM_EXIST
        for x, y in self.__p_obstacle:
            self.__grid[x][y] = ID_GRID_OBSTACLE

        if self.__p_start:
            self.__grid[self.__p_start[0]][self.__p_start[1]] = ID_GRID_START
        if self.__p_goal:
            self.__grid[self.__p_goal[0]][self.__p_goal[1]] = ID_GRID_GOAL

    '''set p_start'''
    def __set_grid_start(self, p_):
        if self.__p_start:
            # p가 item이면 item_empty로 표시
            if self.__p_start in self.__p_item.values():
                self.__grid[self.__p_start[0]][self.__p_start[1]] = ID_GRID_ITEM_EMPTY
            else:
                self.__grid[self.__p_start[0]][self.__p_start[1]] = ID_GRID_FLOOR
        self.__grid[p_[0]][p_[1]] = ID_GRID_START
        return True

    '''set p_goal'''
    def __set_grid_goal(self, p_):
        if self.__p_goal:
            # p가 item이면 item_empty로 표시
            if self.__p_goal in self.__p_item.values():
                self.__grid[self.__p_goal[0]][self.__p_goal[1]] = ID_GRID_ITEM_EMPTY
            else:
                self.__grid[self.__p_goal[0]][self.__p_goal[1]] = ID_GRID_FLOOR
        self.__grid[p_[0]][p_[1]] = ID_GRID_GOAL
        return True

    def __set_grid_reward(self, result_step, cur_y, cur_x, new_y, new_x):
        # new state is out grid
        if result_step == ID_OUT_GRID:
            self.__grid[cur_y][cur_x] = ID_GRID_FLOOR_PASSED
        # new state is goal
        elif result_step == ID_GOAL:
            self.__grid[cur_y][cur_x] = ID_GRID_FLOOR_PASSED
            self.__grid[new_y][new_x] = ID_GRID_ITEM_TAKEN
        # move to obstacle
        elif result_step == ID_OBSTACLE:
            self.__grid[cur_y][cur_x] = ID_GRID_FLOOR_PASSED
        # stay current state
        elif result_step == ID_NOT_MOVE:
            self.__grid[new_y][new_x] = ID_GRID_FLOOR_PASSED
        # new state is start
        elif result_step == ID_RETURN:
            self.__grid[new_y][new_x] = ID_GRID_FLOOR_PASSED
        # 일반적인 이동
        elif result_step == ID_GENERAL_MOVE:
            self.__grid[cur_y][cur_x] = ID_GRID_FLOOR_PASSED
            self.__grid[new_y][new_x] = ID_GRID_FLOOR_CURRENT
        return True
    # endregion GRID

    # region ACTION, REWARD
    '''change x, y by action'''
    def __apply_action(self, action, cur_y, cur_x):
        new_y = cur_y
        new_x = cur_x
        # change x, y
        if action == IDX_ACTION_UP:
            new_y = cur_y - 1
        elif action == IDX_ACTION_DOWN:
            new_y = cur_y + 1
        elif action == IDX_ACTION_LEFT:
            new_x = cur_x - 1
        elif action == IDX_ACTION_RIGHT:
            new_x = cur_x + 1
        else:
            pass
        return int(new_y), int(new_x)

    '''get reward which new state'''
    def __get_reward(self, cur_y, cur_x, new_y, new_x):
        # new state is out grid
        if any([new_y < 0, new_y >= self.height, new_x < 0, new_x >= self.width]):
            # return ID_OUT_GRID, self.REWARD.OUT_GIRD, True
            return ID_OUT_GRID, self.REWARD.OUT_GIRD, False
        # new state is goal
        if self.__p_goal == (new_y, new_x):
            return ID_GOAL, self.REWARD.GOAL, True
        # move to obstacle
        if (new_y, new_x) in self.__p_obstacle:
            return ID_OBSTACLE, self.REWARD.OBSTACLE, True
            # return ID_OBSTACLE, self.REWARD.OBSTACLE, False
        # stay current state
        if (cur_y, cur_x) == (new_y, new_x):
            # return ID_NOT_MOVE, self.REWARD.NOT_MOVE, True
            return ID_NOT_MOVE, self.REWARD.NOT_MOVE, False
        # If items is not goal, it's obstacle
        if (new_y, new_x) in self.__p_item.values():
            return ID_OBSTACLE, self.REWARD.OBSTACLE, True
        # new state is start
        if self.__p_start == (new_y, new_x):
            # return ID_RETURN, self.REWARD.RETURN, True
            return ID_RETURN, self.REWARD.RETURN, False
        # general move
        return ID_GENERAL_MOVE, self.REWARD.MOVE, False
    # endregion ACTION, REWARD
    # endregion PRIVATE


if __name__ == "__main__":
    PATH_LOCAL = '../../'
    oder_train = pd.read_csv(PATH_LOCAL+"data/factory_order_train.csv")
    sim = LogisticEnv(10, 9, PATH_LOCAL+'data/box.csv', PATH_LOCAL+'data/obstacles.csv')

    for epi in range(1):
        row_str = list(oder_train.iloc[epi])[0]
        items = list(set(sim.NAME_ITEM) & set(row_str))
        items.sort()
        sim.set_p_order(items)
        sim.set_route((9, 4), sim.get_p_item(items[0]))

        p_start = sim.reset()
        actions = [IDX_ACTION_UP, IDX_ACTION_LEFT, IDX_ACTION_LEFT, IDX_ACTION_LEFT,
                   IDX_ACTION_UP, IDX_ACTION_UP, IDX_ACTION_UP, IDX_ACTION_UP, IDX_ACTION_UP,
                   IDX_ACTION_LEFT]

        i = 0
        done_ = False
        while done_ is False:
            (new_y, new_x), reward, done_, result_step = sim.step(actions[i])

            grid = sim.get_gird()

            if (done_ is True) or (i == (len(actions) - 1)):
                i = 0
            i += 1



