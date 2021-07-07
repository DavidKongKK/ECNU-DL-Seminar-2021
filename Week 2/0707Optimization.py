import numpy as np
import math

'''

'''

x=[0.0,0.5,0.8,1.1,1.5,1.9,2.2,2.4,2.6,3.0]
x=np.array(x)
y=[0.9,2.1,2.7,3.1,4.1,4.8,5.1,5.9,6.0,7.0]
y=np.array(y)


n=10
w=1.0
b=1.0
lr=0.1
sgd_loss=0.0

def init_param():
    global w,b
    w=0.0
    b=0.0

def calc_loss(y,yhat):
    return (np.sum(np.square(y-yhat)))/(2*n)

def forward():
    yhat=w*x+b
    return yhat

def sgd():
    yhat=[0 for i in range(10)]
    loss=0.0
    global w
    global b
    for j in range(100):
        for i in range(n):
            
            yhat=forward()
            loss=calc_loss(y,yhat)

            dy=2*(y[i]-yhat[i])
            dw=dy*x[i]
            db=dy

            w=w+lr*dw
            b=b+lr*db
            
            print("loss={0}".format(str(loss)))
    return loss

beta1=0.9

def momentum():
    yhat=[0 for i in range(10)]
    
    vdw=0.0
    vdb=0.0
    global w
    global b
    loss=1e8
    i=0
    while loss>sgd_loss:
        i+=1
        yhat=forward()
        loss=calc_loss(y,yhat)

        dy=2*(yhat-y)
        dw=np.dot(dy,x.T)/(1*n)
        db=np.sum(dy,axis=0,keepdims=True)/(1*n)

        vdw=beta1*vdw+(1-beta1)*dw
        vdb=beta1*vdb+(1-beta1)*db

        w=w-lr*vdw
        b=b-lr*vdb
        
        print("loss={0}".format(str(loss)))
    return i

beta2=0.9

lr=0.1
def rmsprop():
    yhat=[0 for i in range(10)]
    
    sdw=0.0
    sdb=0.0

    epsilon=1e-8
    loss=1e10
    i=0
    global w
    global b
    while loss>=sgd_loss:
    #for j in range(100):
        i+=1
        yhat=forward()
        loss=calc_loss(y,yhat)

        dy=2*(yhat-y)
        dw=np.dot(dy,x.T)/(1*n)
        db=np.sum(dy,axis=0,keepdims=True)/(1*n)

        sdw=beta2*sdw+(1-beta2)*pow(dw,2)
        sdb=beta2*sdb+(1-beta2)*pow(db,2)

        w=w-lr*dw/math.sqrt(sdw+epsilon)
        b=b-lr*db/math.sqrt(sdb+epsilon)
        
        print("loss={0}".format(str(loss)))

    return i

beta2=0.999
def adam():
    yhat=[0 for i in range(10)]
    
    vdw,vdb=[0.0,0.0]
    sdw=0.0
    sdb=0.0

    epsilon=1e-8

    loss=1e10

    global w
    global b

    t=0
    while loss>=sgd_loss:
    #for j in range(100):
        t+=1
        yhat=forward()
        loss=calc_loss(y,yhat)

        dy=2*(yhat-y)
        dw=np.dot(dy,x.T)/(1*n)
        db=np.sum(dy,axis=0,keepdims=True)/(1*n)

        vdw=beta1*vdw+(1-beta1)*dw
        vdwc=vdw/(1-pow(beta1,t))

        vdb=beta1*vdb+(1-beta1)*db
        vdbc=vdb/(1-pow(beta1,t))

        sdw=beta2*sdw+(1-beta2)*pow(dw,2)
        sdwc=sdw/(1-pow(beta2,t))

        sdb=beta2*sdb+(1-beta2)*pow(db,2)
        sdbc=sdb/(1-pow(beta2,t))

        w=w-lr*vdwc/math.sqrt(sdwc+epsilon)
        b=b-lr*vdbc/math.sqrt(sdbc+epsilon)
        
        print("loss={0}".format(str(loss)))
        
    return t


sgd_loss=sgd()
init_param()
momentum_epoch=momentum()
init_param()
rmsp_epoch=rmsprop()
init_param()
adam_epoch=adam()
print("sgd_loss={0}\nmomentum_epoch={1}\nrmsp_epoch={2}\nadam_epoch={3}".format(str(sgd_loss),str(momentum_epoch),str(rmsp_epoch),str(adam_epoch)))
