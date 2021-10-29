import infery, numpy as np
model = infery.load(model_path='resnet_dynamic_1_1.pkl', framework_type='trt', inference_hardware='gpu')

inputs = np.random.random((16, 3, 224, 224)).astype('float32')
model.predict(inputs)

model.benchmark(batch_size=16)
