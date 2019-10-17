import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


plane_equation = { 0: lambda p: 2*(p[0,:]+p[1,:])/3+p[2,:],
                   1: lambda p: 2*(-p[0,:]+p[1,:])/3+p[2,:],
                   2: lambda p: -2*(p[0,:]+p[1,:])/3+p[2,:],
                   3: lambda p: 2*(p[0,:]-p[1,:])/3+p[2,:] }
proj_equation = { 0: lambda q: q[0,:]+q[1,:],
                  1: lambda q: -q[0,:]+q[1,:],
                  2: lambda q: -(q[0,:]+q[1,:]),
                  3: lambda q: q[0,:]-q[1,:] }

def classify(p,dec=2):
    corner = np.zeros(p.shape[1])
    polo = np.zeros(p.shape[1])
    corner[np.where((p[0,:]<=0)*(p[1,:]>0))] = 1
    corner[np.where((p[0,:]<0)*(p[1,:]<=0))] = 2
    corner[np.where((p[0,:]>=0)*(p[1,:]<0))] = 3
    polo[np.where(p[2,:]>=1)] = 1
    polo[np.where(p[2,:]<=1)] = 1
    b = np.zeros(p.shape[1])
    for i in range(4):
        b[corner==i] = polo[corner==i]*plane_equation[i](polo[corner==i]*p[:,corner==i])
    b = np.round(b,decimals = dec)
    classification = { 'pole' : polo,
                       'quadrant' : corner,
                       'b' : b }
    return classification

def project_and_scale(p):
    N = p.shape[1]
    classification = classify(p)
    b = classification['b']
    b_set = np.unique(b)
    plane = np.zeros(N)
    for i in range(4):
        tramo = classification['quadrant']==i
        plane[tramo] = proj_equation[i](p[:2,tramo])
    coeff = np.zeros(N)
    height = np.zeros(N)
    for b_elem in b_set:
        coeff[b==b_elem] = np.max(plane[b==b_elem])
        height[b==b_elem] = np.max(p[2,b==b_elem])-1
    coeff[coeff==0]=1
    norm = np.linalg.norm(p[:2,:],axis=0)
    norm[norm==0]=1
    t = (1/coeff)*plane*p[:2,:]/norm
    return t,coeff,height,classification

def stereo_projection(t):
    return np.array([2*t[0,:]/(1+t[0,:]**2+t[1,:]**2),\
                     2*t[1,:]/(1+t[0,:]**2+t[1,:]**2),\
                     (1-t[0,:]**2-t[1,:]**2)/(1+t[0,:]**2+t[1,:]**2)])

def scale_ellipse(p,coeff,height,classification):
    sign = np.array([[1,-1,1,-1],[1,1,-1,-1]])
    for i in range(4):
        tramo = classification['quadrant']==i
        p[0,tramo] = coeff[tramo]*p[0,tramo]
        p[1,tramo] = coeff[tramo]*p[1,tramo]
    p[2,:] = height*p[2,:]
    return p+np.array([[0],[0],[1]])



#v =np.array(p[2,:]>=1)
#p = p[:,v]
#fig = plt.figure()
#ax  = fig.add_subplot(1,1,1,projection='3d')
#ax.scatter(p[0,:],p[1,:],p[2,:],'b')

#q,coeff,height,classification = project_and_scale(p)
#q = stereo_projection(q)
#q = scale_ellipse(q,coeff,height,classification)
#ax.scatter(q[0,:],q[1,:],q[2,:],'r',marker='v')
#plt.show()



