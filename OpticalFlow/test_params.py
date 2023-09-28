from torchinfo import summary
from PWCNet_light import PWCNet
model = PWCNet()
print(summary(model))
print(model)
