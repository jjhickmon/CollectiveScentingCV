import os
import sys
import numpy as np
from matplotlib.widgets import Button
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import utils.image as image_utils
import utils.general as general_utils

class Index(object):
    def __init__(self, min_area, max_area, fig, ax, img_plot, hist_plot, img):
        self.ax = ax
        self.fig = fig
        self.hist_plot = hist_plot
        self.img_plot = img_plot
        self.img = img
        self._increment = 5
        self._area_increment = 10
        self._lower_percent = 10
        self._upper_percent = 90

        self._min_area = min_area
        self._max_area = max_area
        self.upper_area_bound = None

        self.running = True

        self.recompute()

    @property
    def min_area(self):
        min_area = min(self.upper_area_bound, max(0, self._min_area))
        return min_area

    @min_area.setter
    def min_area(self, val):
        self._min_area = min(self.max_area, max(0, val))

    @property
    def max_area(self):
        max_area = min(self.upper_area_bound, max(0, self._max_area))
        return max_area

    @max_area.setter
    def max_area(self, val):
        self._max_area = min(self.upper_area_bound, max(self.min_area, val))

    @property
    def lower_percent(self):
        lower_percent = min(100, max(0, self._lower_percent))
        return lower_percent

    @property
    def upper_percent(self):
        upper_percent = min(100, max(0, self._upper_percent))
        return upper_percent

    @property
    def increment(self):
        increment = min(100, max(0, self._increment))
        return increment

    @property
    def area_increment(self):
        area_increment = min(100, max(0, self._area_increment))
        return area_increment

    @area_increment.setter
    def area_increment(self, val):
        self._area_increment = min(100, max(0, val))

    def checker(self, img, stats, lower_percent=0, upper_percent=100):
        # Filter areas
        # ====================================================
        areas = stats[:,-1]

        condition_1 = areas>np.percentile(areas, lower_percent)
        condition_2 = areas<np.percentile(areas, upper_percent)
        filter_idxs = np.logical_and(condition_1, condition_2)

        filtered_areas = areas[filter_idxs]
        # ====================================================

        # Copy image and make
        show_img = img.copy()
        if len(show_img.shape) == 2:
            show_img = np.tile(show_img[...,np.newaxis], (1,1,3))
        elif show_img.shape[-1] == 1:
            show_img = np.tile(show_img, (1,1,3))

        # Filter the stats on area conditions
        filtered_stats = stats[filter_idxs]

        # Add rectangles to image
        for stat in filtered_stats:
            x, y, w, h, a = stat
            top_left = (x, y)
            bottom_right = (x+w, y+h)
            cv2.rectangle(show_img, top_left, bottom_right, (0,255,0), 10)

        return show_img, areas[filter_idxs]

    def recompute(self):
        img_mask = image_utils.adaptive_thresholding(self.img)
        num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(img_mask)
        show_img, areas = self.checker(self.img, stats, lower_percent=self.lower_percent, upper_percent=self.upper_percent)

        if self.upper_area_bound is None:
            self.upper_area_bound = np.max(areas)

        self.img_plot.set_data(show_img)
        title_str = f"Lower Percentile: {self.lower_percent} -- Upper Percentile: {self.upper_percent}"
        title_str += f"\nIncrement: {self.increment}"
        self.ax[0].set_title(title_str)


        hist_title = f'Min area: {self.min_area} -- Max area: {self.max_area}'
        hist_title += f"\nIncrement: {self.area_increment}"
        self.ax[1].cla()
        self.ax[1].hist(areas)
        self.ax[1].set(xlabel='Area', ylabel='Counts', title=hist_title)

        self.ax[1].axvline(self.min_area, color='r', linestyle='dashed', linewidth=1)
        self.ax[1].axvline(self.max_area, color='b', linestyle='dashed', linewidth=1)

        plt.draw()

    def increase_lower(self, event):
        self._lower_percent += self.increment
        self.recompute()

    def decrease_lower(self, event):
        self._lower_percent -= self.increment
        self.recompute()

    def increase_upper(self, event):
        self._upper_percent += self.increment
        self.recompute()

    def decrease_upper(self, event):
        self._upper_percent -= self.increment
        self.recompute()

    def increase_increment(self, event):
        self._increment += 1
        self.recompute()

    def decrease_increment(self, event):
        self._increment -= 1
        self.recompute()

    def increase_area_increment(self, event):
        self.area_increment += 10
        self.recompute()

    def decrease_area_increment(self, event):
        self.area_increment -= 10
        self.recompute()

    def increase_min_area(self, event):
        self.min_area += self.area_increment
        self.recompute()

    def decrease_min_area(self, event):
        self.min_area -= self.area_increment
        self.recompute()

    def increase_max_area(self, event):
        self.max_area += self.area_increment
        self.recompute()

    def decrease_max_area(self, event):
        self.max_area -= self.area_increment
        self.recompute()

class UserInterface():
    def __init__(self):
        self.min_area = 0
        self.max_area = 1e6

    def setup_vis(self, img):
        fig, ax = plt.subplots(1, 2, figsize=(12,12))
        plt.subplots_adjust(bottom=0.2)
        img_plot = ax[0].imshow(img, cmap='gray')
        hist_plot = ax[1].hist([np.random.randint(5) for _ in range(100)], rwidth=0.95)

        self.callback = Index(self.min_area, self.max_area, fig, ax, img_plot, hist_plot, img)

        self._setup_buttons()
        plt.show()

    def _setup_buttons(self):
        # Buttons
        # ==================================
        button_width = 0.08
        button_height = 0.05

        # Buttons and their placement
        # ------------------------------------------------------------------------
        up_increment   = plt.axes([0.10, 0.20, button_width, button_height])
        down_increment = plt.axes([0.10, 0.15, button_width, button_height])
        increase_lower = plt.axes([0.20, 0.20, button_width, button_height])
        decrease_lower = plt.axes([0.20, 0.15, button_width, button_height])
        increase_upper = plt.axes([0.30, 0.20, button_width, button_height])
        decrease_upper = plt.axes([0.30, 0.15, button_width, button_height])


        up_area_increment   = plt.axes([0.50, 0.10, button_width, button_height])
        down_area_increment = plt.axes([0.50, 0.05, button_width, button_height])
        up_min_area         = plt.axes([0.60, 0.10, button_width, button_height])
        down_min_area       = plt.axes([0.60, 0.05, button_width, button_height])
        up_max_area         = plt.axes([0.80, 0.10, button_width, button_height])
        down_max_area       = plt.axes([0.80, 0.05, button_width, button_height])
        # ------------------------------------------------------------------------

        # Button functionality
        # ------------------------------------------------------------------------

        # Percentiles
        # ==============================
        up_increment_bttn = Button(up_increment, '+ Increment')
        up_increment_bttn.on_clicked(self.callback.increase_increment)

        down_increment_bttn = Button(down_increment, '- Increment')
        down_increment_bttn.on_clicked(self.callback.decrease_increment)

        increase_lower_bttn = Button(increase_lower, '+ Lower %')
        increase_lower_bttn.on_clicked(self.callback.increase_lower)

        decrease_lower_bttn = Button(decrease_lower, '- Lower %')
        decrease_lower_bttn.on_clicked(self.callback.decrease_lower)

        increase_upper_bttn = Button(increase_upper, '+ Upper %')
        increase_upper_bttn.on_clicked(self.callback.increase_upper)

        decrease_upper_bttn = Button(decrease_upper, '- Upper %')
        decrease_upper_bttn.on_clicked(self.callback.decrease_upper)

        # Area
        # ==============================
        up_area_increment_bttn = Button(up_area_increment, '+ Increment')
        up_area_increment_bttn.on_clicked(self.callback.increase_area_increment)

        down_area_increment_bttn = Button(down_area_increment, '- Increment')
        down_area_increment_bttn.on_clicked(self.callback.decrease_area_increment)

        up_min_area_bttn = Button(up_min_area, '+ MinArea')
        up_min_area_bttn.on_clicked(self.callback.increase_min_area)

        down_min_area_bttn = Button(down_min_area, '- MinArea')
        down_min_area_bttn.on_clicked(self.callback.decrease_min_area)

        up_max_area_bttn = Button(up_max_area, '+ MaxArea')
        up_max_area_bttn.on_clicked(self.callback.increase_max_area)

        down_max_area_bttn = Button(down_max_area, '- MaxArea')
        down_max_area_bttn.on_clicked(self.callback.decrease_max_area)
        # ------------------------------------------------------------------------

    def run(self, img, img_dir, load_previous=False):
        annotations_dir = general_utils.make_directory('UI_annotation_history', root=img_dir)
        self.results_path = os.path.join(annotations_dir, 'UI_results.json')

        if load_previous:
            try:
                with open(self.results_path, 'r') as infile:
                    min_max_results = json.load(infile)
                self.min_area = min_max_results['min_area']
                self.max_area = min_max_results['max_area']
            except:
                self.setup_vis(img)
        else:
            self.setup_vis(img)
            self.min_area = self.callback.min_area
            self.max_area = self.callback.max_area
