from tensorflow.keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rescale=1./255)
g = gen.flow_from_directory('data/train', target_size=(224,224), batch_size=1)
print("class_indices:", g.class_indices)
