import os
import sys
import numpy as np

import cv2
import json
import utils.image as image_utils
import utils.general as general_utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "individual" : (0,255,0),
    "cluster"    : (188, 19, 254)
}

class Visualizer():
    def __init__(self, img, prev_gt_points=None):
        self.img = img
        self.prev_gt_points = prev_gt_points

        self.data_points = []
        self.hover_circle = None
        self.hover_size = 8

        self.circle_size = 5.5
        self._init_class()
        self.close_plot = False
        self._start()

    def _init_class(self):
        self.class_idx = 0
        self.classes = list(CLASS_COLORS.keys())
        self.current_class = self.classes[self.class_idx]
        self.circle_color = np.array(CLASS_COLORS[self.current_class])/255.0
        self.hover_color = np.array(CLASS_COLORS[self.current_class])/255.0

    def _update_class(self):
        self.class_idx = (self.class_idx + 1) % 2
        self.current_class = self.classes[self.class_idx]
        self.circle_color = np.array(CLASS_COLORS[self.current_class])/255.0
        self.hover_color = np.array(CLASS_COLORS[self.current_class])/255.0

    def _start(self):
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12,12))

        # Draw image
        self.ax.imshow(self.img, cmap='gray')
        self._show_keys()
        # Setup events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.fig.canvas.mpl_connect("motion_notify_event", self.mouseover)

        # Try to load datapoints
        self._load_datapoints()

        # Keep plot open and end it nicely when onclose called.
        while True:
            if plt.waitforbuttonpress(0):
                if self.close_plot:
                    break

    def _load_datapoints(self):
        if self.prev_gt_points is not None:
            try:
                for x, y, class_label in self.prev_gt_points:
                    color = np.array(CLASS_COLORS[class_label])/255.0
                    circle = self._add_circle((x,y), color=color)
                    self.data_points.append((circle, (x,y), class_label))
            except:
                print("\nFailed to load. File is corrupt.")

    def get_points(self):
        points = [(ele[1], ele[2]) for ele in self.data_points]
        return points

    def onclose(self):
        self.close_plot = True
        plt.close(self.fig)

    def _show_keys(self):
        cmds = "Keyboard Commands"
        cmds += "\n==========="
        cmds += '\np: Load annotations'
        cmds += '\nr: Remove all labels'
        # cmds += '\nc: Chance class:'
        # cmds += '    Cluster (Green)'
        # cmds += '    Individual (Red)'
        cmds += '\nu: Undo previous label'
        # cmds += '\n]: Increase Size'
        # cmds += '\n[: Decrease Size'
        cmds += '\nq: Quit/Finish'
        self.ax.text(-500, 100, cmds, fontsize=10)

    def onkey(self, event):
        if event.key == 'r':
            while self.data_points:
                point, data, class_label = self.data_points.pop()
                point.remove()
        elif event.key == 'p':
            self._load_datapoints()
        elif event.key == 'u':
            if self.data_points:
                prev_circle, prev_point, prev_class_label = self.data_points.pop()
                prev_circle.remove()
        elif event.key == ']':
            self.hover_size += 1
        elif event.key == '[':
            self.hover_size -= 1
        elif event.key == 'c':
            self._update_class()
        elif event.key == 'q':
            self.onclose()
            return

        self.refresh()

    def refresh(self):
        plt.draw()

    def _add_circle(self, point, color):
        circle = plt.Circle(point, self.circle_size, color=color)
        self.ax.add_artist(circle)
        return circle

    def onclick(self, event):
        point = (event.xdata, event.ydata)
        circle = self._add_circle(point, self.circle_color)
        self.data_points.append((circle, point, self.current_class))

        self.refresh()

    def mouseover(self, event):
        point = (event.xdata, event.ydata)
        if self.hover_circle is not None:
            self.hover_circle.remove()
        self.hover_circle = plt.Circle(point, self.hover_size, color=self.hover_color)
        self.ax.add_artist(self.hover_circle)
        self.refresh()

class Search():
    def __init__(self, img):
        self.img = img

    def filter_by_area(self, img, stats, min_area, max_area):
        areas = stats[:,-1]
        total_img_area = int(np.product(img.shape[:2]))
        max_permitable_detection_area = total_img_area * 1/8

        if min_area is None:
            min_area = np.min(areas)
        if max_area is None:
            max_area = np.max(areas)

        condition_1 = areas > min_area
        condition_2 = areas < max_area

        filter_idxs = np.logical_and(condition_1, condition_2)
        return filter_idxs, max_permitable_detection_area

    def run(self, min_area_i, max_area_i, gt_points, target_label, opening_filter=(9,9)):
        # img_mask = image_utils.adaptive_filter_plus_opening(self.img, kernel_dim=opening_filter, invert=True)
        img_mask = image_utils.adaptive_filter_plus_opening(self.img, kernel_dim=opening_filter, invert=True)
        num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(img_mask)

        filter_idxs, max_permitable_detection_area = self.filter_by_area(self.img, stats, min_area_i, max_area_i)
        filtered_stats = stats[filter_idxs]

        num_targets = 0
        num_found = 0
        for stat_i, (x1, y1, w, h, a) in enumerate(filtered_stats):
            bbox_area = w*h
            if bbox_area > max_permitable_detection_area:
                continue

            x2, y2 = x1+w, y1+h
            update_gt_points = gt_points.copy()
            for g_i, ((gt_x, gt_y), label) in enumerate(gt_points):
                if label != target_label:
                    continue
                else:
                    num_targets += 1
                if x1 < gt_x < x2 and y1 < gt_y < y2:
                    try:
                        num_found += 1
                        del update_gt_points[g_i]
                    except:
                        continue

            gt_points = update_gt_points
        num_targets = max(1, num_targets)
        accuracy = num_found / num_targets * 100
        #print(f"\nNum Found: {num_found}/{num_targets} -- Accuracy: {accuracy:0.4f}")
        return accuracy

class UserInterface():
    def __init__(self):
        self.evaluate_groups = True
        stepsize = 100

        self.min_areas = list(range(100, 700, stepsize))
        self.max_areas = list(range(800, 2000, stepsize))

    def eval_individual_detections(self, ground_truth_points):
        min_maxs = []
        individual_accuracies = []
        for min_area in self.min_areas:
            for max_area in self.max_areas:
                min_maxs.append((min_area, max_area))
                individual_accuracy = self.search.run(min_area, max_area, ground_truth_points, target_label='individual')
                individual_accuracies.append(individual_accuracy)
        return individual_accuracies, min_maxs

    def eval_cluster_detections(self, min_maxs, ground_truth_points):
        group_accuracies = []
        for min_area, max_area in min_maxs:
            group_accuracy = self.search.run(max_area, None, ground_truth_points, target_label='cluster')
            group_accuracies.append(group_accuracy)
        return group_accuracies

    def _run_search(self, top_N=5):
        ground_truth_points = self.vis.get_points()

        with open(self.gt_points_path, 'w') as outfile:
            print(f"Saving GT points to '{self.gt_points_path}' ")
            # ground_truth_points_save = [(float(x), float(y), label) for (x,y), label in ground_truth_points]
            ground_truth_points_save = []
            for (x,y), label in ground_truth_points:
                try:
                    ground_truth_points_save.append((float(x), float(y), label))
                except:
                    continue

            json.dump(ground_truth_points_save, outfile)

        # Run individual accuracies
        print("Running experiments to find optimal bounding box area parameters...")
        individual_accuracies, min_maxs = self.eval_individual_detections(ground_truth_points)

        # Run group accuracies
        if self.evaluate_groups:
            group_accuracies = self.eval_cluster_detections(min_maxs, ground_truth_points)

            accuracies = [sum(ele) for ele in list(zip(individual_accuracies, group_accuracies))]
        else:
            accuracies = individual_accuracies

        # Sort
        sorted_idxs = np.argsort(accuracies)[::-1][:top_N]
        sorted_accs = np.array(accuracies)[sorted_idxs]
        sorted_min_maxs = np.array(min_maxs)[sorted_idxs]

        # Display top
        # print(f"Top {top_N}")
        # print("-----------")
        # for acc, (min_area, max_area) in zip(sorted_accs, sorted_min_maxs):
        #     print(f"Min: {min_area} -- Max: {max_area} -- Acc: {acc:0.4f}%")

        # Select best
        self.min_area = np.mean(np.array(sorted_min_maxs)[:,0])
        self.max_area = np.mean(np.array(sorted_min_maxs)[:,1])
        print(f"Best min (average): {self.min_area} -- Best max (average): {self.max_area}")

        with open(self.results_path, 'w') as outfile:
            json.dump({"min_area" : self.min_area, "max_area" : self.max_area}, outfile)

    def _load_previous_points(self):
        prev_gt_points = None
        try:
            if os.path.exists(self.gt_points_path):
                print(f"Reading GT points from '{self.gt_points_path}' ")
                with open(self.gt_points_path, 'r') as infile:
                    prev_gt_points = json.load(infile)
        except:
            print(f"Unable to load points from '{self.gt_points_path}' ")
            prev_gt_points = None
        return prev_gt_points

    def run(self, img, img_dir, load_previous=True):
        annotations_dir = general_utils.make_directory('UI_annotation_history', root=img_dir)
        self.results_path = os.path.join(annotations_dir, 'UI_results.json')
        self.gt_points_path = os.path.join(annotations_dir, 'UI_annotations.json')

        prev_gt_points = self._load_previous_points()

        # Instantiate helpers
        self.vis = Visualizer(img, prev_gt_points)
        self.search = Search(img)

        if load_previous:
            try:
                with open(self.results_path, 'r') as infile:
                    min_max_results = json.load(infile)
                self.min_area = min_max_results['min_area']
                self.max_area = min_max_results['max_area']
            except:
                self._run_search()
        else:
            self._run_search()
