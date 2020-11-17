"""
Question 2 - Exercise 1
Warming up!
"""


# Represents a box and its contents
class Box:

    def __init__(self, items, p_box):
        self.items = items  # {k:v}
        self.p_box = p_box

    # calculate probability of retrieving an item by type
    def calc_prob(self, item):
        if self.items[item]:
            return self.items[item] / sum(self.items.values())
        return 0  # not an item


# Boxes: Red (r), Green (g), and Blue (b)
r = Box({"apple": 3, "orange": 4, "lime": 3}, 0.2)
g = Box({"apple": 3, "orange": 3, "lime": 4}, 0.6)
b = Box({"apple": 1, "orange": 1, "lime": 0}, 0.2)

# Probability of selecting an apple
# p(apple) = Σ_boxes [p(apple|box)*p(box)]
p_apple = r.calc_prob("apple") * r.p_box +\
          g.calc_prob("apple") * g.p_box +\
          b.calc_prob("apple") * b.p_box
print("(a) Probability of selecting an apple: {0:1.2f}".format(p_apple))

# Probability that the green box was picked given an orange was selected
# Using Bayes Theorem: p(g_b|o) = (p(o|g_b) * p(g_b)) / p(o)
p_orange = r.calc_prob("orange") * r.p_box +\
           g.calc_prob("orange") * g.p_box +\
           b.calc_prob("orange") * b.p_box

p_green_box_given_orange = g.calc_prob("orange") * g.p_box / p_orange
print("(b) Probability that an orange came from the green box given an orange" +\
      " was selected: {0:1.2f}".format(p_green_box_given_orange))