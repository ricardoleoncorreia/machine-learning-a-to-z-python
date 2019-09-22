# -*- coding: utf-8 -*-
import math

def calculate_upper_bound(round_number, rewards_total, selections_total):
    if (selections_total > 0):
        average_reward = rewards_total / selections_total
        delta_i = math.sqrt(3/2 * math.log(round_number) / selections_total)
        return average_reward + delta_i
    return 1e400