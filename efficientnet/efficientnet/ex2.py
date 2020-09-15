from efficientnet.tfkeras import EfficientNetB0
import numpy as np

image = np.random.randint(0, 255, (1, 224, 224, 3)) / 255.
label = np.array([1])

model = EfficientNetB0()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(image, label, epochs = 1)

pred = model.predict(image)
print(pred)