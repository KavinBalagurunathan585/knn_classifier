import math
import csv
from collections import Counter
import matplotlib.pyplot as plt

def load_data(filename): #load data from csv file
    data=[]
    with open (filename,'r') as f:
        reader=csv.DictReader(f)
        for r in reader:
            if r['age']=='' or r['sex']=='' or r['pclass']==''or r['survived']=='':
                continue
            try:
                survived=int(r['survived'])
                age=float(r['age'])
                pclass=int(r['pclass'])
                if r['sex']=="female":
                    sex=1
                else:
                    sex=0
            except(ValueError,TypeError):
                continue
            data.append((pclass,sex,age,survived))
    return data
    
def mean_std(train,index): # find mean and std
    val=[]
    mean_dif=[]
    sum1=0
    
    total=0
    for i in train:
        val.append(i[index])
    for v in val:
        total=total+v
    mean=total/len(val)
    for i in val:
        mean_dif.append((i-mean)**2)
    for i in mean_dif:
        sum1=sum1+i
    variance=sum1/len(val)
    std=math.sqrt(variance)
    return mean,std

def normalize(data):  # use z test
    val1=(data[0]-mean_class)/std_class
    val2=data[1]
    val3=(data[2]-mean_age)/std_age
    val4=data[3]
    return (val1,val2,val3,val4)

data=load_data("Titanic.csv")
split_index=int(len(data)*0.7)
train=data[:split_index]
test=data[split_index:]
mean_class,std_class=mean_std(train,0)
mean_age,std_age=mean_std(train,2)
train_data=[]
test_data=[]
for i in train:
    train_data.append(normalize(i))
for i in test:
    test_data.append(normalize(i))

def distance(a,b):  #calculate distance using euclidean formula
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def knn(query,k): #calculate label that appears most
    dist=[]
    k_neighbour=[]
    for i in train_data:
        d=distance(i[:3],query)
        dist.append((d,i[3]))
    dist.sort()
    for _,label in dist[:k]:
        k_neighbour.append(label)
    count=Counter(k_neighbour)
    max_neighbour=count.most_common(1)[0][0]
    return max_neighbour
TN=FN=FP=TP=0
for i in test_data:
    actual=i[3]
    predicted=knn(i[:3],17)
    if predicted==1 and actual==1:
        TP+=1
    elif predicted==1 and actual==0:
        FP+=1
    elif predicted==0 and actual==0:
        TN+=1
    elif predicted==0 and actual==1:
        FN+=1
if TP+FN>0:
    recall=TP / (TP + FN)
else:
    recall=0
if TP+FP>0:
    precision=TP/(TP+FP)
else:
    precision=0
if precision+recall>0:
    f1score=2*((precision*recall)/(precision+recall))
else:
    precision=0
if (TP + TN + FP + FN) > 0:
    accuracy = (TP + TN) / (TP + TN + FP + FN)
else:
    accuracy = 0
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
metrics={"Recall":recall,"precision":precision,"accuracy":accuracy,"f1score":f1score}

metrics = {
  'Accuracy': accuracy,
  'Precision': precision,
  'Recall': recall,
  'F1': f1score
}
 # plot bar graph
plt.figure(figsize=(5,5))
plt.bar(metrics.keys(),metrics.values(),color=['red','blue','green','orange'])
plt.xticks(fontsize=12)
plt.ylim(0,1)
plt.title("Model Evaluation Metrics")
plt.ylabel("Metric value")
plt.show()
