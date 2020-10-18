import numpy as np
import pandas as pd
import os
import tensorflow as tf
import pickle
from sklearn.model_selection import StratifiedKFold
import model
from efficientnet import tfkeras
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost

cwd = os.getcwd()
train_path = os.path.join(cwd,"train")
test_path = os.path.join(cwd,"test")
train_label_file = os.path.join(cwd,"train.csv")

def image_resize(path, resize_path):
    from PIL import Image
    files = os.listdir(path)
    for file in files:
        image_contents = Image.open(os.path.join(path, file))
        image_contents = image_contents.resize((224,224))
        image_contents.save(os.path.join(resize_path, file),'JPEG')

train_resize_path = os.path.join(cwd,"train_resize")
test_resize_path = os.path.join(cwd, "test_resize")
# image_resize(train_path,train_resize_path)
# image_resize(test_path,test_resize_path)

train_path = os.path.join(cwd,"train_resize")
test_path = os.path.join(cwd, "test_resize")

def load_data(read_path, save_path, label_file=None,is_train=False):

    files = os.listdir(read_path)

    data = list()
    label = list()
    if is_train and label_file is not None:
        label_data = pd.read_csv(label_file)
        print(label_data["label"].value_counts())
    for file in files:
        if is_train:
            label.append(label_data[label_data["filename"].astype(str)==str(file)]["label"].values[0])

        image_contents = tf.read_file(os.path.join(read_path,file))
        image = tf.image.decode_jpeg(image_contents,channels=3)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            img = sess.run(image)
            data.append(img)
    res = {"data":data,"files":files}
    if is_train:
        res.update({"label":label})
    with open(save_path,"wb") as f:
        pickle.dump(res,f)

train_data_file = os.path.join(cwd,"train_data.pkl")
test_data_file = os.path.join(cwd,"test_data.pkl")
# load_data(train_path,train_data_file,train_label_file,is_train=True)
# load_data(test_path,test_data_file)


data = pickle.load(open(train_data_file,"rb"))
train_data, train_label = np.array(data["data"])/255., data["label"]
test_data = pickle.load(open(test_data_file,"rb"))
test_files = test_data["files"]
test_data = np.array(test_data["data"])/255.


print("train data shape: ",train_data.shape)
print("test data shape: ",test_data.shape)

width, height, channel = train_data[0].shape


def convertLabel(data):
    labels = np.unique(data).tolist()
    print("unique label number: ",len(labels))
    str2id = dict()
    for i in range(len(labels)):
        str2id[labels[i]] = i
    with open("label2id.txt","w") as f:
        for key, value in str2id.items():
            tmp = str(key) + " " + str(value)
            f.write(tmp)
            f.write("\n")
    label_vec = list()
    for i in range(len(data)):
        label_vec.append(str2id[data[i]])
    with open("./labels.pkl","wb") as f:
        pickle.dump(label_vec,f)
    return label_vec

def one_hot(x, num_class=6):
    y = np.zeros((len(x), num_class))
    for i in range(len(x)):
        y[i, x[i]] = 1.0
    return y
# label = convertLabel(train_label)

label = pickle.load(open("./labels.pkl","rb"))
# print(label)

print(train_data[0].shape)

def label_smooth(x,e=0.1):
    v = x.shape[0]
    x = (1-e)*x+e/v
    return x


# 构建训练过程

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True

n = 5
kfold = StratifiedKFold(n, shuffle=True)

train_data = np.array(train_data)
label = np.array(label)

# # use test data and presudo_label as training
# # presudo_label
# data_presudo = pd.read_csv("./result/test_presudo_label.csv")
# test_presudo_label = data_presudo["label_mean"].values
# train_data = np.append(train_data, test_data,axis=0)
# label = np.append(label,test_presudo_label,axis=0)
# print("train data shape after presudo label: ", train_data.shape)
# print("train label shape after presudo label: ", label.shape)

X_train = train_data.copy()
Y_train = label.copy()

# test_data_prob = np.zeros((n,test_data.shape[0],6))
# test_data_prob_xgb = np.zeros((n,test_data.shape[0],6))

def dataGenerator(x, y, batch=32,alpha=0.2):
    from random import shuffle
    length = x.shape[0]
    arr = [i for i in range(length)]
    steps = length // batch + 1
    while True:
        shuffle(arr)
        for i in range(steps):

            if alpha > 0.:
                if i == steps-1:
                    weight = np.random.beta(alpha,alpha,length-batch*i)
                    x_weight = weight.reshape(length-batch*i,1,1,1)
                    y_weight = weight.reshape(length-batch*i,1)
                    index = np.random.randint(steps)
                    yield (x[arr[i*batch:(i+1)*batch]]*x_weight + x[arr[index*(length-batch*i):(index+1)*(length-batch*i)]]*(1-x_weight), y[arr[i*batch:(i+1)*batch]]*y_weight+y[arr[index*(length-batch*i):(index+1)*(length-batch*i)]]*(1-y_weight))

                else:
                    weight = np.random.beta(alpha, alpha, batch)
                    x_weight = weight.reshape(batch,1,1,1)
                    y_weight = weight.reshape(batch,1)
                    index = np.random.randint(steps-1)

                    yield (x[arr[i*batch:(i+1)*batch]]*x_weight + x[arr[index*batch:(index+1)*batch]]*(1-x_weight), y[arr[i*batch:(i+1)*batch]]*y_weight+y[arr[index*batch:(index+1)*batch]]*(1-y_weight))

            else:
                yield (x[arr[i*batch:(i+1)*batch]], y[arr[i*batch:(i+1)*batch]])


for index, (train_index, valid_index) in enumerate(kfold.split(train_data,label)):

    print("========= %d =========="%index)
    x_train, x_valid, y_train, y_valid = X_train[train_index], X_train[valid_index], Y_train[train_index], Y_train[
        valid_index]

    y_train_ml = one_hot(y_train)
    # y_train_ml = label_smooth(y_train_ml)  # label smoothing

    y_valid_ml = one_hot(y_valid)

    # # data Augment
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True
    # )
    # datagen.fit(x_train)



    batch_size = 32
    steps = x_train.shape[0]//batch_size + 1

    # 使用预训练模型efficientNet
    efficientNet = tfkeras.EfficientNetB3(
                include_top=False,
                weights='imagenet',
                input_shape=(width, height,channel)
        )
    globalAP = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAP")
    x_in = tf.keras.layers.Input(shape=(width, height, channel))
    x = x_in
    x = efficientNet(x)
    x = globalAP(x)
    x = tf.keras.layers.Dense(512,activation="relu",kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256,activation="relu",kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(128,activation="relu",kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(6,activation="softmax")(x)

    md = tf.keras.Model(x_in,x)

    checkpoint_path = "training/index-%d-weights-{epoch:02d}-{val_acc:.5f}.ckpt"%index
    checkpoint_dir = os.path.dirname(checkpoint_path)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    md.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

    call_backs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       min_lr=0.0000001,
                                                       cooldown=1,
                                                       verbose=1,),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                    save_best_only=True,
        #                                    monitor="val_acc",
        #                                    verbose=1,
        #                                    mode="max",
        #                                    save_weights_only=True,
        #                                    ),
                  ]
    md.fit_generator(dataGenerator(x_train, y_train_ml, batch=batch_size,alpha=0.2),steps_per_epoch=steps,epochs=180,verbose=2,callbacks=call_backs,validation_data=(x_valid,y_valid_ml),workers=4,)
    # md.fit_generator(datagen.flow(x_train, y_train_ml, batch_size=batch_size),steps_per_epoch=steps,epochs=180,verbose=2,callbacks=call_backs,validation_data=(x_valid,y_valid_ml),workers=4,)

    #
    # # # 加载最新模型
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # md.load_weights(latest)

    # 评估测试集
    # test_data_prob[index,:,:] = md.predict(test_data,batch_size=test_data.shape[0])
    # test_data_prob = md.predict(test_data, batch_size=test_data.shape[0])
    # with open("./test_pred_effnet3_mixup_%d.pkl"%index, "wb") as f:
    #     pickle.dump(test_data_prob, f)

    # # # =============== 插入xgboost
    # # print(globalAP.output_shape)
    #
    # get_GlobalAveragePooling = tf.keras.backend.function([x_in,
    #                                                       tf.keras.backend.learning_phase()],
    #                                                      [globalAP.output])
    # def dataSplit(data):
    #
    #     new_data = np.zeros((data.shape[0], 1280))  # new_data = np.zeros((data.shape[0], globalAP.output_shape[1]))
    #     for j in range(10):
    #         if j == 9:
    #             new_data[(data.shape[0])//10*j:,:] = get_GlobalAveragePooling([data[(data.shape[0])//10*j:],0])[0]
    #         else:
    #             new_data[j*(data.shape[0])//10:(j+1)*(data.shape[0])//10,:] = get_GlobalAveragePooling([data[j*(data.shape[0])//10:(j+1)*(data.shape[0])//10],0])[0]
    #     return new_data
    # xgb_train_data[index, :,:1280] = dataSplit(x_train)
    # xgb_train_data[index,:,-1] = y_train
    # xgb_valid_data[index,:,:1280] = dataSplit(x_valid)
    # xgb_valid_data[index,:,-1] = y_valid
    # xgb_test_data[index,:,:] = dataSplit(test_data)


    # xgb_params = {'learning_rate': 0.1, "n_estimators":1000,"max_depth":5,'min_child_weight': 1, 'seed': 27,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 0.5,"tree_method":"gpu_hist"}
    # clf = xgboost.XGBClassifier(**xgb_params)
    # clf.fit(xgb_x_train,y_train)
    # xgb_x_valid = dataSplit(x_valid)
    # y_valid_pred = clf.predict(xgb_x_valid)
    # print("valid data accuracy: ",accuracy_score(y_valid,y_valid_pred))
    #
    # xgb_test_train = get_GlobalAveragePooling([test_data,0])[0]
    # test_data_prob_xgb[index,:,:] = clf.predict_proba(xgb_test_train)

# # 存储用于xgboost的数据
# xgb_data = {"xgb_train_data":xgb_train_data,"xgb_valid_data":xgb_valid_data,"xgb_test_data":xgb_test_data}
# with open(r"./xgb_data.pkl","wb") as f:
#     pickle.dump(xgb_data,f)

# test_data_prob = test_data_prob_xgb

# # mean
# def meanFunciton(data):
#     assert isinstance(data,np.ndarray)
#     assert len(data.shape) == 3
#
#     tmp_pred_prob = data.mean(axis=0)
#     tmp_pred = tmp_pred_prob.argmax(axis=1)
#     return tmp_pred
#
# mean_pred = meanFunciton(test_data_prob)
#
# # 读取评估
#
# test_pred = np.zeros((n,test_data.shape[0]))
# for i in range(n):
#     test_pred[i,:] = np.argmax(test_data_prob[i,:,:],axis=1)
#
# # 获取标签对应的名称
# id2label = dict()
# with open("label2id.txt","r") as f:
#     for line in f.readlines():
#         tmp = line.strip().split()
#         id2label[int(tmp[1])] = str(tmp[0])
# # voting
# voting_pred = list()
# for i in range(test_data.shape[0]):
#     tmp = test_pred[:,i]
#     count_dict = {}
#     for j in tmp:
#         if j in count_dict.keys():
#             count_dict[j] += 1
#         else:
#             count_dict[j] = 1
#     voting_pred.append(list(sorted(count_dict.items(),key=lambda x:x[1],reverse=True))[0][0])
#
#
#
# ## 获取测试集的索引
# df_test = pd.DataFrame(test_files,columns=["filename"])
# df_test["label_voting"] = voting_pred
# df_test["labelname_voting"] = df_test["label_voting"].map(lambda x:id2label[x])
# df_test["label_mean"] = mean_pred
# df_test["labelname_mean"] = df_test["label_mean"].map(lambda x:id2label[x])
# df_test["index"] = df_test["filename"].map(lambda x:int(x.strip().split(".")[0]))
#
# df_test = df_test.sort_values(by="index")
# print(df_test)
# df_test.to_csv("test_pred.csv",index=False)
#
# arr_mean = df_test[["index","labelname_mean"]].values
# with open("./test_pred_mean.csv","w") as f:
#     for l in arr_mean:
#         tmp = str(l[0])+","+str(l[1])
#         f.write(tmp)
#         f.write("\n")
#
# arr_vote = df_test[["index","labelname_voting"]].values
# with open("./test_pred_voting.csv","w") as f:
#     for l in arr_vote:
#         tmp = str(l[0])+","+str(l[1])
#         f.write(tmp)
#         f.write("\n")
#




























