import torch
from torchvision import models
from fuzzy_math import DEVICE, fuzzy_min, fuzzy_max

class FuzzyArt:

    """
    ART Neural Network

    Parameters:
        cp:        Choice Parameter
        vp:        Vigilance Parameter
        lr:        Learning Rate

    """

    def __init__(self, cp, vp, lr):
        self.cp = cp
        self.vp = vp
        self.lr = lr

        self.categories = {} # key: Category, value: Weight Vector of Category

    def create_category(self, input):
        category = len(self.categories)
        self.categories[category] = torch.ones(len(input)).to(DEVICE)
        self.update_category(category, input)

    def update_category(self, category, input):
        self.categories[category] = self.lr * fuzzy_min(input, self.categories[category]) + (1 - self.lr) * self.categories[category]

    def compare(self, input):
        comp_vals = {} # key: Category, value: Comparison Value of Category
        for category in self.categories:
            fuzzy_arr = fuzzy_min(input, self.categories[category])
            comp_vals[category] = torch.sum(fuzzy_arr) / (self.cp + torch.sum(self.categories[category]))
        return comp_vals

    def choose(self, comp_vals):
        max_category = -1
        max_comp_val = -1
        for category in comp_vals:
            comp_val = comp_vals[category]
            if comp_val > max_comp_val:
                max_category = category
                max_comp_val = comp_val
        return max_category

    def recognize(self, input, category, comp_vals, training):
        fuzzy_arr = fuzzy_min(input, self.categories[category])
        similarity = torch.sum(fuzzy_arr) / torch.sum(self.categories[category])
        if similarity >= self.vp:
            # Resonate
            if training:
                self.update_category(category, input)
            return True
        else:
            # Reset
            comp_vals[category] = -1
            return False
            
    def fuzzy_art(self, input, training=True):
        # Complement-coding Normalization
        input = torch.hstack((input, 1-input))

        comp_vals = self.compare(input)
        recognized = False
        while not recognized:
            choice = self.choose(comp_vals)
            if choice == -1:
                if training:
                    choice = self.create_category(input)
                break
            else:
                recognized = self.recognize(input, choice, comp_vals, training)
        print(len(self.categories))
        return choice
    