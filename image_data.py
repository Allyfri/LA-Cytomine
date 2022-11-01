# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2017. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__          = "Vanhee Laurent <laurent.vanhee@student.uliege.ac.be>"
__copyright__       = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"

import os
import sys
import csv
import config
import user_data
from dictionary_data import parse_positions, parse_ref_annotations, parse_annotations, parse_annotation_actions
from gazemap import cluster_points, score_user_on_image, generate_reduced_heatmap
from pygazeanalyser.gazeplotter import make_heatmap, save_heatmap, draw_raw, draw_scanpath
from PIL import Image
import numpy as np
import gc
import time
import datetime
from gazemap import annotation_order
from bs4 import BeautifulSoup

class Image_data:
    # class containg data related to 1 particular image


    def __init__(self, project_name, image_dir, manager, user_list):
        """
        Creates an Image data object containing image info and position info
        for all the users that have positions in a dictionary. It loads
        :param project_name: gold/silver
        :param image_dir: image_xxxxx
        :param manager: data_manager object (keep link for when needed)
        """

        # init all variables
        self.manager = manager
        self.image_id = image_dir.split('_')[1]
        self.image_dir = config.WORKING_DIRECTORY + project_name + "/images/" + image_dir + "/"
        self.positions_dir = self.image_dir + "user_positions/"
        self.annotations_dir = self.image_dir + "user_annotations/"
        self.ref_annotation_dir = self.image_dir + "reference_cytomine_annotations.csv"
        self.annotation_actions_dir = self.image_dir + "user_actions/"
        self.user_positions = {}
        self.user_annotations = {}
        self.user_descriptions = {}
        self.user_actions = {}
        self.ref_annotations = None
        self.gaussians = {'zoom_4': None,
                          'zoom_5': None,
                          'zoom_6': None,
                          'zoom_7': None,
                          'zoom_8': None,
                          'zoom_9': None,
                          'zoom_10': None,
                          'zoom_11': None}
        self.image = Image.open(self.image_dir + "image.png")
        self.rescaled_width, self.rescaled_height = self.image.size
        self.user_data = {}

        end = long(1000 * time.mktime(datetime.datetime.strptime(config.exam_time, "%Y-%m-%d %H:%M:%S").timetuple()))

        # load positions to memory
        u_positions_files = os.listdir(self.positions_dir)
        for pos in u_positions_files:
            if user_list is None or pos.split("_")[0] in user_list:
                f = open(self.positions_dir + pos, 'rb')
                pos_id = pos.split('_')[0]
                csv_in = csv.reader(f)
                data = list(csv_in)
                data.pop(0)
                f.close()
                pos_data = parse_positions(data, self, duration=20, calc_gauss=True, end_date=end)
                self.user_positions[pos_id] = pos_data

        # loads ref annotations to memory
        try:
            f = open(self.ref_annotation_dir, 'rb')
        except:
            f = None
        if f is not None:
            csv_in = csv.reader(f)
            data = list(csv_in)
            data.pop(0)
            f.close()
            ann_data = parse_ref_annotations(data)
            self.ref_annotations = ann_data

        # loads annotation actions to memory
        u_action_files = os.listdir(self.annotation_actions_dir)
        for pos in u_action_files:
            if user_list is None or pos.split("_")[0] in user_list:
                f = open(self.annotation_actions_dir + pos, 'rb')
                pos_id = pos.split('_')[0]
                csv_in = csv.reader(f)
                data = list(csv_in)
                data.pop(0)
                f.close()
                action_data = parse_annotation_actions(data, self.user_positions[pos_id], self.ref_annotations, end_date=end)
                self.user_actions[pos_id] = action_data

        # loads user annotations to memory
        u_annotation_files = os.listdir(self.annotations_dir)
        for annot in u_annotation_files:
            if user_list is None or annot.split("_")[0] in user_list:
                f = open(self.annotations_dir + annot, 'rb')
                user_id = annot.split('_')[0]
                csv_in = csv.reader(f)
                data = list(csv_in)
                data.pop(0)
                f.close()
                ann_data = parse_annotations(data)
                self.user_annotations[user_id] = ann_data

        # loads descriptions to memory
        for annot in u_annotation_files:
            if user_list is None or annot.split("_")[0] in user_list:
                descriptions = []
                media_in_descriptions = []
                user_id = annot.split('_')[0]
                times = []
                for j in range(len(self.user_annotations[user_id]['description_timestamp'])) :
                    if self.user_annotations[user_id]['description_timestamp'][j] > 0 :
                       description = self.user_annotations[user_id]['description'][j]

                       #clean description from <a>, <img> and <video> tag
                       soup = BeautifulSoup(description,features="html.parser")
                       description = soup.get_text(separator=' ')
                       # break into lines and remove leading and trailing space on each
                       lines = (line.strip() for line in description.splitlines())
                       # break multi-headlines into a line each
                       chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                       # drop blank lines
                       description = '\n'.join(chunk for chunk in chunks if chunk)

                       #and count these tags
                       medias_nb = len(soup.findAll('img'))
                       medias_nb = medias_nb + len(soup.findAll('a'))
                       medias_nb = medias_nb + len(soup.findAll('video'))

                       descriptions.append(description)
                       media_in_descriptions.append(medias_nb)

                       times.append(self.user_annotations[user_id]['timestamp'][j])
                self.user_descriptions[user_id] = {}
                self.user_descriptions[user_id]['description'] = descriptions
                self.user_descriptions[user_id]['timestamp'] = times
                self.user_descriptions[user_id]['media'] = media_in_descriptions


        self.zoom_max = self.max_zoom()

    def init_user_data_link(self, user_data):
        """
        creates links between images and users
        :param user_data: list of user_data objects
        :return: None
        """
        for user in user_data:
            if user.user_id in self.user_positions:
                self.user_data[user.user_id] = user
                user.image_data[self.image_id] = self

    def generate_heatmap(self, user_id):
        """
        Generates a heatmap associated to this image and a user, and keeps it in self
        :param user_id: user to generate heatmap with
        :return: None
        """
        pos = self.user_positions[user_id]
        pos['heatmap'] = make_heatmap(pos, (self.rescaled_width, self.rescaled_height), self)


    def generate_all_heatmaps(self):
        """
        Generates heatmaps for all users associated to this image
        :return: None
        """
        for u_id in self.user_positions:
            self.generate_heatmap(u_id)

    def remove_heatmap(self, user_id):
        """
        removes a heatmap from memory
        :param user_id: user id of heatmap to be removed
        :return: None
        """
        pos = self.user_positions[user_id]
        heatmap = pos['heatmap']
        pos['heatmap'] = None
        del heatmap

    def remove_all_heatmaps(self):
        """
        Remove all heatmaps associated to this image
        :return: None
        """
        for u_id in self.user_positions:
            self.remove_heatmap(u_id)
        gc.collect()

    def save_all_heatmaps_by_image(self):
        """
        Saves all heatmaps in image files, this method compares all heatmaps associated to this image
        and outputs images based on the minimums and maximums from all heatmaps. It also applies a logarithmic
        normalizer because of some very high values compared to others which affects scaling
        :return: None
        """
        rgb_im = self.image.convert('RGB')
        rgb_im.save('converted_image.jpg')
        dir = self.image_dir + "gazemaps_image_method/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        max_val = 0
        avg_val = 0
        l = 0
        # determines an average value for all the ln heatmaps
        # determines the highest value found on all the heatmaps
        for u_id in self.user_positions:
            pos = self.user_positions[u_id]
            heatmap = np.copy(pos['heatmap'])
            heatmap = heatmap + 1
            heatmap = np.log10(heatmap)
            tmp = np.max(heatmap)
            max_val = max(tmp, max_val)
            if len(heatmap[heatmap > 0]) > 0:
                avg_val = np.mean(heatmap[heatmap > 0])
                l = l + 1
            del heatmap
        if l == 0:
            os.remove('converted_image.jpg')
            return

        avg_val = avg_val/l

        # Save all heatmaps while taking to account max and avg
        for u_id in self.user_positions:
            out = dir + u_id + "_heatmap.png"
            pos = self.user_positions[u_id]
            heatmap = np.copy(pos['heatmap'])
            heatmap = heatmap + 1
            heatmap = np.log10(heatmap)
            heatmap[0][0] = max_val
            save_heatmap(heatmap, (self.rescaled_width, self.rescaled_height), imagefile='converted_image.jpg', savefilename=out, alpha=0.5, avg=avg_val, annotations=self.ref_annotations)
            del heatmap
        gc.collect()
        os.remove('converted_image.jpg')

    def save_all_heatmaps_ln(self):
        """
        Saves all heatmaps in image files. It applies a logarithmic
        normalizer because of some very high values compared to others which affects scaling
        :return: None
        """
        rgb_im = self.image.convert('RGB')
        rgb_im.save('converted_image.jpg')
        dir = self.image_dir + "gazemaps_ln_method/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        for u_id in self.user_positions:
            out = dir + u_id + "_heatmap.png"
            pos = self.user_positions[u_id]
            heatmap = np.copy(pos['heatmap'])
            heatmap = heatmap + 1
            heatmap = np.log10(heatmap)
            save_heatmap(heatmap, (self.rescaled_width, self.rescaled_height), imagefile='converted_image.jpg', savefilename=out, alpha=0.5, annotations=self.ref_annotations)
            del heatmap
        gc.collect()
        os.remove('converted_image.jpg')

    def save_all_heatmaps(self):
        """
        Save all heatmaps in images files, no normalization
        :return: None
        """
        rgb_im = self.image.convert('RGB')
        rgb_im.save('converted_image.jpg')
        dir = self.image_dir + "gazemaps_basic/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        for u_id in self.user_positions:
            out = dir + u_id + "_heatmap.png"
            pos = self.user_positions[u_id]
            heatmap = np.copy(pos['heatmap'])
            save_heatmap(heatmap, (self.rescaled_width, self.rescaled_height), imagefile='converted_image.jpg', savefilename=out, alpha=0.5, annotations=self.ref_annotations)
            del heatmap
        gc.collect()
        os.remove('converted_image.jpg')

    def save_all_raw(self, bar):
        """
        Saves all positions in an image file, each position is represented by a dot on the image.
        :return: None
        """
        rgb_im = self.image.convert('RGB')
        rgb_im.save('converted_image.jpg')
        del rgb_im
        dir = self.image_dir + "raw_points_images/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        for u_id in self.user_positions:
            out = dir + u_id + "_points.png"
            pos = self.user_positions[u_id]
            draw_raw(pos, (self.rescaled_width, self.rescaled_height), imagefile='converted_image.jpg', savefilename=out)
            bar.next()
        #gc.collect()
        os.remove('converted_image.jpg')

    def save_all_scanpath(self, bar):
        """
        Saves all scanpaths, due to the number of positions, it applies clustering methods to have a better and more viewable image
        :return: None
        """
        rgb_im = self.image.convert('RGB')
        rgb_im.save('converted_image.jpg')
        dir = self.image_dir + "scanpath_images/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        for u_id in self.user_positions:
            out = dir + u_id + "_scanpath.png"
            pos = self.user_positions[u_id]
            tmp = cluster_points(pos)
            draw_scanpath(tmp, (self.rescaled_width, self.rescaled_height), imagefile='converted_image.jpg', savefilename=out)
            bar.next()
        gc.collect()
        os.remove('converted_image.jpg')

    def save_all_heatmaps_reduced(self, bar):
        rgb_im = self.image.convert('RGB')
        rgb_im.save('converted_image.jpg')
        dir = self.image_dir + "gazemap_reduced/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        for u_id in self.user_positions:
            out = dir + u_id + "_heatmap.png"
            heatmap = generate_reduced_heatmap(self.user_positions[u_id],(self.rescaled_width, self.rescaled_height), self)
            save_heatmap(heatmap, (self.rescaled_width, self.rescaled_height), imagefile='converted_image.jpg',
                         savefilename=out, alpha=0.5, annotations=self.ref_annotations)
            gc.collect()
            bar.next()
        os.remove('converted_image.jpg')


    def score_users(self, u_list):
        """
        Scores the users in relation to this image, each user is given a score on "how well" they viewed the image
        :param u_list: list of user_ids
        :return: list of user scores
        """
        ret = []
        ann_ret = []
        a_empty = None
        if self.ref_annotations is not None:
                a_empty = [0 for i in range(len(self.ref_annotations['x']))]
        for u in u_list:
            if u in self.user_positions and u in self.user_actions:
                s, a = score_user_on_image(self.user_positions[u], self.user_actions[u], self)
                ret.append(s)

            elif u in self.user_positions and u not in self.user_actions:
                s, a = score_user_on_image(self.user_positions[u], None, self)
                ret.append(s)
            else:
                a = None
                ret.append(0)

            if self.ref_annotations is not None:
                if a is not None:
                    ann_ret.append(a)
                else:
                    ann_ret.append(a_empty)

        return ret, ann_ret

    def annotation_order(self, ann1, ann2, user_list):

        ret = []
        for user in user_list:
            u_id = user.user_id
            if u_id not in self.user_positions:
                ret.append(0)
            else:
                val = annotation_order(self.user_positions[u_id], self.ref_annotations, ann1, ann2, self.gaussians, 10, self.zoom_max)
                ret.append(val)

        return ret


    def max_zoom(self):
        """
        gets the highest level of zoom reached by a student
        :return: [0-10] zoom value
        """
        max_z = 0
        for u in self.user_positions:
            tmp = self.user_positions[u]['zoom']
            if len(tmp) > 0:
                max_z = max(max_z, np.max(tmp))
        return max_z

    def nb_ref_annotations(self):
        """
        Returns the number of reference annotations
        :return: nb
        """
        if self.ref_annotations is None:
            return 0
        else:
            return len(self.ref_annotations['x'])

    def nb_user_annotations(self):
        """
        Returns the number of user annotations
        :return: nb
        """
        if self.user_annotations is None:
            return 0
        else:
            sum = 0
            for u in self.user_annotations:
                sum = sum + len(u['x'])
            return sum

    def nb_of_users(self):
        """
        Returns the number of users who visited this image
        :return: nb
        """
        return len(self.user_data)

    def nb_descriptions(self):
        """
        Returns the number of non empty description
        :return: nb
        """
        if self.user_descriptions is None:
            return 0
        else:
            sum = 0
            for u in self.user_descriptions:
                sum = sum + len(u['description'])
            return sum

    def aggr_info_descriptions(self):
        """
        Returns the number of non empty description
        :return: nb
        """
        exit(1)
        ret = {}
        if self.user_annotations is None:
            return ret
        else:
            descriptions = []
            for u in self.user_descriptions:
                descriptions.append(u['description'])
            #total nb of word
            total_description_word = sum(list(map(lambda x: len(x.split()), description)))
            #total nb of char
            total_description_char = sum(list(map(lambda x: len(x), description)))
            #avg ng of word
            total_description_word/self.nb_descriptions()
            #avg nb of char
            total_description_char/self.nb_descriptions()
            #median nb of word
            #median nb of char
            if self.nb_descriptions() == 0:
                return 0 # pour les 2

            # put them in an array
            chars = np.zeros(self.nb_descriptions())
            words = np.zeros(self.nb_descriptions())
            i = 0
            for desc in descriptions():
                chars[i] = len(desc)
                words[i] = len(desc.split())
                i += 1
            # sort them
            positions = np.sort(positions)
            chars[i] = np.sort(chars)
            words[i] = np.sort(words)

            # return position number in the middle (median)
            chars[np.int(self.nb_descriptions()/2)]
            words[np.int(self.nb_descriptions()/2)]

            return ret


    def __repr__(self):
        return "image :" + str(self.image_id) + ", pos :" + str(len(self.user_positions)) + ", links :" + str(len(self.user_data)) + "\n"

    def __str__(self):
        return "image :" + str(self.image_id) + ", pos :" + str(len(self.user_positions)) + ", links :" + str(len(self.user_data)) + "\n"



