import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost
from sklearn.model_selection import StratifiedKFold


n = 5
kfold = StratifiedKFold(n,shuffle=True)



# 读取训练、验证、测试数据
path = "./xgb_data.pkl"
with open(path,"rb") as f:
    data = pickle.load(f)


train_data = data["xgb_train_data"]
X_train = train_data[:,:1280]
Y_train = train_data[:,-1]

X_test = data["xgb_test_data"]
test_pred_prob = np.zeros((n,X_test.shape[0],6))  # 预测概率


xgb_params = {'learning_rate': 0.1, "n_estimators":2000,"max_depth":5,'min_child_weight': 1, 'seed': 27,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 0.5,"tree_method":"gpu_hist"}

val_acc = np.zeros((n))
for index, (train_index, valid_index) in enumerate(kfold.split(X_train,Y_train)):
    x_train, y_train, x_valid, y_valid = X_train[train_index], Y_train[train_index], X_train[valid_index], Y_train[valid_index]

    def train(x,y,params):
        xgb = xgboost.XGBClassifier(**params)
        xgb.fit(x,y)
        return xgb

    def pred(x,clf):
        y_pred = clf.predict(x)
        y_pred_prob = clf.predict_proba(x)
        return y_pred, y_pred_prob

    clf = train(x_train,y_train,xgb_params)
    y_train_pred, _ = pred(x_train,clf)
    y_valid_pred, _ = pred(x_valid,clf)
    _, test_pred_prob[index,:,:] = pred(X_test,clf)

    print("train accuracy: ",accuracy_score(y_train,y_train_pred),"valid accuracy: ",accuracy_score(y_valid,y_valid_pred))
    val_acc[index] = accuracy_score(y_valid,y_valid_pred)

print("valid accuracy: ",np.mean(val_acc))

test_data_prob = test_pred_prob
# 读取测试文件名称
test_data_file = "./test_data.pkl"
test_data = pickle.load(open(test_data_file,"rb"))
test_files = test_data["files"]

# mean
def meanFunciton(data):
    assert isinstance(data,np.ndarray)
    assert len(data.shape) == 3

    tmp_pred_prob = data.mean(axis=0)
    tmp_pred = tmp_pred_prob.argmax(axis=1)
    return tmp_pred

mean_pred = meanFunciton(test_data_prob)

# 读取评估

test_pred = np.zeros((n,X_test.shape[0]))
for i in range(n):
    test_pred[i,:] = np.argmax(test_data_prob[i,:,:],axis=1)

# 获取标签对应的名称
id2label = dict()
with open("label2id.txt","r") as f:
    for line in f.readlines():
        tmp = line.strip().split()
        id2label[int(tmp[1])] = str(tmp[0])
# voting
voting_pred = list()
for i in range(X_test.shape[0]):
    tmp = test_pred[:,i]
    count_dict = {}
    for j in tmp:
        if j in count_dict.keys():
            count_dict[j] += 1
        else:
            count_dict[j] = 1
    voting_pred.append(list(sorted(count_dict.items(),key=lambda x:x[1],reverse=True))[0][0])



## 获取测试集的索引
df_test = pd.DataFrame(test_files,columns=["filename"])
df_test["label_voting"] = voting_pred
df_test["labelname_voting"] = df_test["label_voting"].map(lambda x:id2label[x])
df_test["label_mean"] = mean_pred
df_test["labelname_mean"] = df_test["label_mean"].map(lambda x:id2label[x])
df_test["index"] = df_test["filename"].map(lambda x:int(x.strip().split(".")[0]))

df_test = df_test.sort_values(by="index")
print(df_test)
df_test.to_csv("test_pred_eff0_xgb.csv",index=False)

arr_mean = df_test[["index","labelname_mean"]].values
with open("./test_pred_eff0_xgb_mean.csv","w") as f:
    for l in arr_mean:
        tmp = str(l[0])+","+str(l[1])
        f.write(tmp)
        f.write("\n")

arr_vote = df_test[["index","labelname_voting"]].values
with open("./test_pred_eff0_xgb_voting.csv","w") as f:
    for l in arr_vote:
        tmp = str(l[0])+","+str(l[1])
        f.write(tmp)
        f.write("\n")
