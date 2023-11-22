import numpy as np
from copy import copy

pi = np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine_mmgp8(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]

    """
    # Longitudes (en metros)

    # Matrices DH (completar)
    T0 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    T1 = dh(q[0],0,0,pi/2)
    T2 = dh(0.330+0.076+0.01, q[1], 0.04, pi/2)
    T3 = dh(0,-q[2]+pi/2, 0.345, 0)
    T4 = dh(0,q[3],0.04,pi/2)
    T5 = dh(0.340,-q[4],0,-pi/2)
    T6 = dh(0,q[5],0,pi/2)
    T7 = dh(0.08,q[6],0,0)
    # Efector final con respecto a la base
    T = T0@T1@T2@T3@T4@T5@T6@T7
    #@T2@T3@T4@T5@T6@T7
    return T


def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6, q7]
    """
    # Alocacion de memoria
    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine_mmgp8(q)
    # Iteracion para la derivada de cada columna
    for i in range(7):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Calcular nuevamenta la transformacion homogenea e
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+delta)
        T_inc = fkine_mmgp8(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0:3,i]=(T_inc[0:3,3]-T[0:3,3])/delta
        dq[i] = dq[i] - delta
    return J


def ikine_mmgp8(xdes, q0):
   """
   Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0.
   Emplear el metodo de newton
   """
   epsilon  = 0.001
   max_iter = 1000
   delta    = 0.00001
   # Files for the logs
   error = open("/home/user/lab_ws/errorac.txt", "w")    

   q  = copy(q0)
   for i in range(max_iter):
       # Main loop
       J = jacobian_position(q)
       f = fkine_mmgp8(q)
       e = xdes - f[0:3,3]
       q = q + np.dot(J.T, e)
       error.write(str(e[0])+' '+str(e[1]) +' '+str(e[2])+'\n')
       if (np.linalg.norm(e) < epsilon):
           break
       pass
   return q

def ik_gradient_mmgp8(xdes, q0):
   """
   Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0.
   Emplear el metodo gradiente
   """
   epsilon  = 0.001
   max_iter = 1000
   delta    = 0.001
   alpha = 0.1


   q  = copy(q0)
   for i in range(max_iter):
       # Main loop
       J = jacobian_position(q)
       f = fkine_mmgp8(q)
       e = xdes - f[0:3,3]
       q = q + alpha*np.dot(J.T, e)
      
       if (np.linalg.norm(e) < epsilon):
           break
          
       pass
  
   return q
