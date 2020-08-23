"""
@author: Maziar Raissi

"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize
import latex
from scipy.interpolate import griddata
import time
import meshio
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import tensorflow_probability as tfp
np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t, u, v, layers):
        
        X = np.concatenate([x, y, t], 1)
        Y = np.concatenate([u, v], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        
        self.u = u
        self.v = v
        
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.dtype = "float32"
        # Descriptive Keras model 
        #print("weights",self.weights)
        # tf.keras.backend.set_floatx(self.dtype)
        # self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        # self.model.add(tf.keras.layers.Lambda(
        #     lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
        #         # Initialize NN
        # for width in layers[1:-1]:
        #     self.model.add(tf.keras.layers.Dense(
        #         width, activation=tf.nn.tanh,
        #         kernel_initializer="glorot_normal"))
        # self.model.add(tf.keras.layers.Dense(
        #         layers[-1], activation=None,
        #         kernel_initializer="glorot_normal"))
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        
        # tf placeholders and graph
        self.x = tf.Variable(X[:,0:1],dtype=tf.float32)
        self.y = tf.Variable(X[:,1:2],dtype=tf.float32)
        self.t = tf.Variable(X[:,2:3],dtype=tf.float32)
              
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x, self.y, self.t)

        
        self.u = tf.Variable(u,dtype=tf.float32)
        self.v = tf.Variable(v,dtype=tf.float32)     
        self.Y = Y 

                    
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                          method = 'L-BFGS-B', 
        #                                                          options = {'maxiter': 50000,
        #                                                                     'maxfun': 50000,
        #                                                                     'maxcor': 50,
        #                                                                     'maxls': 50,
        #                                                                     'ftol' : 1.0 * np.finfo(float).eps})        
        
                 

        
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            Xtmp=tf.concat([x,y,t], 1)
            psi_and_p = self.neural_net(Xtmp,self.weights,self.biases)
            
            psi = psi_and_p[:,0:1]
            p = psi_and_p[:,1:2]
            u = tape.gradient(psi, y) 
            #print("u :",np.shape(u))
            v = -tape.gradient(psi, x)
            #print("v :",np.shape(v))
            
            u_t = tape.gradient(u, t)  
            #print("u_t :",np.shape(u_t))
            u_x = tape.gradient(u, x)
            #print("u_x :",np.shape(u_x))
            u_y = tape.gradient(u, y)
            #print("u_y :",np.shape(u_y))
            
            u_xx = tape.gradient(u_x, x)
            #print("u_xx :",np.shape(u_xx))
            u_yy = tape.gradient(u_y, y) 
            #print("u_yy :",np.shape(u_yy))
            
            v_t = tape.gradient(v, t)
            #print("v_t :",np.shape(v_t))
            v_x = tape.gradient(v, x) 
            #print("v_x :",np.shape(v_x))
            v_y = tape.gradient(v, y)
            #print("v_y :",np.shape(v_y))
            v_xx = tape.gradient(v_x, x)  
            #print("v_xx :",np.shape(v_xx))
            v_yy = tape.gradient(v_y, y)
            #print("v_yy :",np.shape(v_yy))
            p_x = tape.gradient(p, x)
            #print("p_x :",np.shape(p_x))
            p_y = tape.gradient(p, y)
            #print("p_y :",np.shape(p_y))
    
            f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
            f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
            del tape

        return u, v, p, f_u, f_v
    def lambda12(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            Xtmp=tf.concat([x,y,t], 1)
            psi_and_p = self.neural_net(Xtmp,self.weights,self.biases)
        
            psi = psi_and_p[:,0:1]
            p = psi_and_p[:,1:2]
            u = tape.gradient(psi, y) 
            #print("u :",np.shape(u))
            v = -tape.gradient(psi, x)
            #print("v :",np.shape(v))
            
            u_t = tape.gradient(u, t)  
            #print("u_t :",np.shape(u_t))
            u_x = tape.gradient(u, x)
            #print("u_x :",np.shape(u_x))
            u_y = tape.gradient(u, y)
            #print("u_y :",np.shape(u_y))
            
            u_xx = tape.gradient(u_x, x)
            #print("u_xx :",np.shape(u_xx))
            u_yy = tape.gradient(u_y, y) 
            #print("u_yy :",np.shape(u_yy))
            
            v_t = tape.gradient(v, t)
            #print("v_t :",np.shape(v_t))
            v_x = tape.gradient(v, x) 
            #print("v_x :",np.shape(v_x))
            v_y = tape.gradient(v, y)
            #print("v_y :",np.shape(v_y))
            v_xx = tape.gradient(v_x, x)  
            #print("v_xx :",np.shape(v_xx))
            v_yy = tape.gradient(v_y, y)
            #print("v_yy :",np.shape(v_yy))
            p_x = tape.gradient(p, x)
            #print("p_x :",np.shape(p_x))
            p_y = tape.gradient(p, y)
            #print("p_y :",np.shape(p_y))
            fu1 = (u*u_x + v*u_y)
            fu2 = (u_xx + u_yy) 
            fu3 = u_t + p_x
            fv1 = (u*v_x + v*v_y)
            fv2 = (v_xx + v_yy)
            fv3 = v_t + p_y 
            # f_u =  self.lambda_1*fu1 - lambda_2*fu2+fu3
            # f_v =  self.lambda_1*fu1 - lambda_2*fu2+fu3
            del tape

        return fu1,fu2,fu3,fv1,fv2,fv3 
    
    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))
        
    def grad(self, X, Y):
        with tf.GradientTape() as tape:
            loss_value = self.lossval
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads
    
    def train(self, nIter): 
        start_time = time.time()
        trainable=[self.x, self.y, self.t,self.u, self.v, self.lambda_1, self.lambda_2]
        
        for it in range(nIter):
            with tf.GradientTape(persistent=True) as tape:
             [fu1,fu2,fu3,fv1,fv2,fv3] = self.lambda12(self.x,self.y,self.t)
             loss = lambda:tf.reduce_sum(tf.square(self.u - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v - self.v_pred)) + \
                    tf.reduce_sum(tf.square(fu1*self.lambda_1+fu2*self.lambda_2+fu3)) + \
                    tf.reduce_sum(tf.square(fv1*self.lambda_1+fv2*self.lambda_2+fv3))      
             lossval = tf.reduce_sum(tf.square(self.u - self.u_pred)) + \
                       tf.reduce_sum(tf.square(self.v - self.v_pred)) + \
                       tf.reduce_sum(tf.square(self.f_u_pred)) + \
                       tf.reduce_sum(tf.square(self.f_v_pred))           
             grads = tape.gradient(lossval,trainable)       
             optimizer_Adam = tf.keras.optimizers.Adam()
             optimizer_Adam.apply_gradients(zip(grads, trainable))
             optimizer_Adam.minimize(loss,trainable)
             del tape
                # Print

             if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = lossval
                lambda_1_value = self.lambda_1
                lambda_2_value = self.lambda_2
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()

        # optimizer_results=tfp.optimizer.lbfgs_minimize(
        #     self.loss, 
        #     initial_position=np.random.randn(dim),f_relative_tolerance=1.0 * np.finfo(float).eps,
        #     max_iterations=50000,tolerance=1e-08
        #     )
        
        # print(optimizer_results)
        # print("TTTTTTTT",[lambda_1_value,lambda_2_value,self.x_tf,self.y_tf, self.t_tf])
        
        # scipy.optimize.minimize(fun=self.loss,x0=[self.sess.run(self.lambda_1),self.sess.run(self.lambda_2),self.sess.run(self.x_tf), self.sess.run(self.y_tf), self.sess.run(self.t_tf)],
        #                         method='l-bfgs-b',options = {'maxiter': 50000,
        #                                                      'maxfun': 50000,
        #                                                      'maxcor': 50,
        #                                                      'maxls': 50,
        #                                                      'ftol' : 1.0 * np.finfo(float).eps})
                
        
            
    
    def predict(self, x_star, y_star, t_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star

def plot_solution(X_star, u_star, index):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
if __name__ == "__main__": 
    N_train = 500
    
    layers = [3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2]
    
    # Load Data
   # ——————————————————————————————————————————————————————————————————
    mesh = meshio.read(
    filename='/Users/howardxu/work/neuro-net/Physics-informed-neural-networks/main/Data/rec000068.vtu', 
    # string, os.PathLike, or a buffer/open file
      # optional if filename is a path; inferred from extension
    )    
    
    x=mesh.points[:,0]
    y=mesh.points[:,1]
    t=np.arange(0,68*1.467451833203625E-004,1.467451833203625E-004);

    u=mesh.point_data['flds1']#x
    v=mesh.point_data['flds2']#y
    p=mesh.point_data['flds3']#pressure
    N = x.shape[0]
    T = t.shape[0]
   # ——————————————————————————————————————————————————————————————————    

    x=x.flatten()[:,None]
    y=y.flatten()[:,None]
    t=t.flatten()[:,None]
    XX = np.tile(x, (1,T)) # N x T
    YY = np.tile(y, (1,T)) # N x T
    TT = np.tile(t, (N,1)) # N x T
    UU = np.tile(u, (1,T))
    VV = np.tile(v, (1,T))
    PP = np.tile(p, (1,T))
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT 
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(50)
    t=np.arange(0,68*1.467451833203625E-004,1.467451833203625E-004);
    TT = np.tile(t, (N,1))
    # Test Data
    snap = np.array([10])
    x_star = XX[:,snap]
    y_star = YY[:,snap]
    t_star = TT[:,snap]
    
    u_star = UU[:,snap]
    v_star = VV[:,snap]
    p_star = PP[:,snap]
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    
    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))                  
    
    # Plot Results
#    plot_solution(X_star, u_pred, 1)
#    plot_solution(X_star, v_pred, 2)
#    plot_solution(X_star, p_pred, 3)    
#    plot_solution(X_star, p_star, 4)
#    plot_solution(X_star, p_star - p_pred, 5)
     
    #Import again in case of override
    x1=mesh.points[:,0]
    y1=mesh.points[:,1]
    # Predict for plotting
    X_star=np.concatenate([x1.flatten()[:,None],y1.flatten()[:,None]],axis=1)
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    # noise = 0.01        
    # u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    # v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])    

    # # Training
    # model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    # model.train(1000)
        
    # lambda_1_value_noisy = model.sess.run(model.lambda_1)
    # lambda_2_value_noisy = model.sess.run(model.lambda_2)
      
    # error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    # error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100
        
    # print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
    # print('Error l2: %.5f%%' % (error_lambda_2_noisy))     

             
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
     # Load Data
    data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')
           
    x_vort = data_vort['x'] 
    y_vort = data_vort['y'] 
    w_vort = data_vort['w'] 
    modes = np.asscalar(data_vort['modes'])
    nel = np.asscalar(data_vort['nel'])    
    
    xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
    yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
    ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')
    
    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0, 2.0])
    
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    ####### Row 0: Vorticity ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    
    
    ####### Row 1: Training data ##################
    ########      u(t,x,y)     ###################        
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0],  projection='3d')
    ax.axis('off')

    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]
    
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    
    ########      v(t,x,y)     ###################        
    ax = plt.subplot(gs1[:, 1],  projection='3d')
    ax.axis('off')
    
    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]
    
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    
    # savefig('./figures/NavierStokes_data') 

    
    fig, ax = newfig(1.015, 0.8)
    ax.axis('off')
    
    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ########### 
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Predicted pressure', fontsize = 10)
    
    ########     Exact p(t,x,y)     ########### 
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Exact pressure', fontsize = 10)
    
    
    ######## Row 3: Table #######################
    gs3 = gridspec.GridSpec(1, 2)
    gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs3[:, :])
    ax.axis('off')
    
    s = r'$\begin{tabular}{|c|c|}';
    s = s + r' \hline'
    s = s + r' Correct PDE & $\begin{array}{c}'
    s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
    s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' \end{tabular}$'
 
    ax.text(0.015,0.0,s)
    
    # savefig('./figures/NavierStokes_prediction') 