from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from preprocessing import *

def create_submission(score, test, prediction):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('activity_id,outcome\n')
    total = 0
    for id in test['activity_id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

## 读取数据
train, test, features = read_test_train()

gbm = xgb.Booster(model_file='redhat_bussiness_verone_20181130.model')

score = '0.9974'
test_prediction = gbm.predict(xgb.DMatrix(test[features]))

create_submission(score, test, test_prediction)