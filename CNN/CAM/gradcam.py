import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K

def main(): 
    model = ResNet50(weights='imagenet')
    model.summary()

    img_path = 'images/sample.jpg'
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=5)[0])
    
    top = np.argmax(preds[0])
    output = model.output[:, top]
    last_conv_layer = model.get_layer('add_16')
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (255 * heatmap).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    cv2.imwrite('images/sample_output.jpg', superimposed_img)


if __name__ == '__main__':
    main()