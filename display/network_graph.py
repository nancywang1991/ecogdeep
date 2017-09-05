from keras.utils.visualize_util import plot
from keras.models import Model, load_model 


model = load_model("/home/wangnxr/models/ecog_vid_model_lstm_c95_itr_0_t_3900_.h5")
plot(model, to_file="/home/wangnxr/ecogvidmodel.png", show_shapes=True)
plot(model, to_file="/home/wangnxr/ecogvidmodel.svg", show_shapes=True)

