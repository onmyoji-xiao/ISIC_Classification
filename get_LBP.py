import numpy as np
import cv2
from pylab import *


class LBP:
    def __init__(self):
        self.revolve_map = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7, 15: 8, 17: 9, 19: 10, 21: 11, 23: 12,
                            25: 13, 27: 14, 29: 15, 31: 16, 37: 17, 39: 18, 43: 19, 45: 20, 47: 21, 51: 22, 53: 23,
                            55: 24,
                            59: 25, 61: 26, 63: 27, 85: 28, 87: 29, 91: 30, 95: 31, 111: 32, 119: 33, 127: 34, 255: 35}

        self.uniform_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 12: 8,
                            14: 9, 15: 10, 16: 11, 24: 12, 28: 13, 30: 14, 31: 15, 32: 16,
                            48: 17, 56: 18, 60: 19, 62: 20, 63: 21, 64: 22, 96: 23, 112: 24,
                            120: 25, 124: 26, 126: 27, 127: 28, 128: 29, 129: 30, 131: 31, 135: 32,
                            143: 33, 159: 34, 191: 35, 192: 36, 193: 37, 195: 38, 199: 39, 207: 40,
                            223: 41, 224: 42, 225: 43, 227: 44, 231: 45, 239: 46, 240: 47, 241: 48,
                            243: 49, 247: 50, 248: 51, 249: 52, 251: 53, 252: 54, 253: 55, 254: 56,
                            255: 57}

    def describe(self, img_path, width, height):
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        image_array = np.array(img_gray)

        return image_array

    def calute_basic_lbp(self, image_array, i, j):
        sum = []
        if image_array[i - 1, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i - 1, j] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i - 1, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)

        return sum

    def get_min_for_revolve(self, arr):
        values = []
        circle = arr
        circle.extend(arr)
        for i in range(0, 8):
            j = 0
            sum = 0
            bit_num = 0
            while j < 8:
                sum += circle[i + j] << bit_num
                bit_num += 1
                j += 1
            values.append(sum)
        return min(values)

    def calc_sum(self, r):
        num = 0
        while (r):
            r &= (r - 1)
            num += 1
        return num

    def lbp_basic(self, image_array):

        basic_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                sum = self.calute_basic_lbp(image_array, i, j)
                bit_num = 0
                result = 0

                for s in sum:
                    result += s << bit_num
                    bit_num += 1
                basic_array[i, j] = result

        return basic_array

    # Get LBP rotation-invariant pattern features

    def lbp_revolve(self, image_array):
        revolve_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                sum = self.calute_basic_lbp(image_array, i, j)
                revolve_key = self.get_min_for_revolve(sum)
                revolve_array[i, j] = self.revolve_map[revolve_key]
        return revolve_array

    # Get LBP Equivalent Mode Features
    def lbp_uniform(self, image_array):
        uniform_array = np.zeros(image_array.shape, np.uint8)
        basic_array = self.lbp_basic(image_array)
        width = image_array.shape[0]
        height = image_array.shape[1]

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                k = basic_array[i, j] << 1
                if k > 255:
                    k = k - 255
                xor = basic_array[i, j] ^ k

                num = self.calc_sum(xor)

                if num <= 2:
                    uniform_array[i, j] = self.uniform_map[basic_array[i, j]]
                else:
                    uniform_array[i, j] = 58
        return uniform_array

    # Get LBP rotation-invariant equivalent mode features
    def lbp_revolve_uniform(self, image_array):
        uniform_revolve_array = np.zeros(image_array.shape, np.uint8)
        basic_array = self.lbp_basic(image_array)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                k = basic_array[i, j] << 1
                if k > 255:
                    k = k - 255
                xor = basic_array[i, j] ^ k
                num = self.calc_sum(xor)
                if num <= 2:
                    uniform_revolve_array[i, j] = self.calc_sum(basic_array[i, j])
                else:
                    uniform_revolve_array[i, j] = 9
        return uniform_revolve_array

    def get_lbph(self, image_array, cell_width, cell_height):
        width = image_array.shape[0]
        height = image_array.shape[1]
        numx = int(width / cell_width)
        numy = int(height / cell_height)

        feature = []
        for i in range(numy):
            for j in range(numx):
                cell = image_array[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
                cell_uniform = self.lbp_uniform(cell)
                hist = cv2.calcHist([cell_uniform], [0], None, [59], [0, 59])
                hist = cv2.normalize(hist, hist).flatten()
                feature.append(hist)
        return feature
