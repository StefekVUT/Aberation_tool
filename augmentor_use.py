import Augmentor
p = Augmentor.Pipeline("C:/Users/smocko/Desktop/aberation_tool/data/augmentor")
p.rotate(0.7, 25, 25)
p.zoom(0.3, 1.1, 1.6)

p.sample(20)
