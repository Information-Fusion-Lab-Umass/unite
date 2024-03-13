import numpy as np
import scipy.special as special


def unite_estimator_fast(n, p, y, A, z, beta, TTE=None):
  treated_neighb = A.dot(z)
  control_neighb = A.dot(1-z)

  w_vec = np.zeros(n)
  for r in range(beta+1):
    for k in range(r+1):
      temp = special.binom(treated_neighb, k) * special.binom(control_neighb, r-k)
      temp1 = ((((1-p)**(k)) * ((-p)**(r-k)))/( p**r) ) - ((((p-1)**(k)) * ((p)**(r-k)))/( (1-p)**r))
      w_vec += temp*temp1

  est = w_vec.dot(y)
  if TTE is None:
    return est/n



def unite_estimator_dr_fast(n, p, y, A, z, beta, TTE=None,outcom=None):
  treated_neighb = A.dot(z)
  control_neighb = A.dot(1-z)
  
  outz = outcom*z
  y_n = y - outz

  w_vec = np.zeros(n)
  for r in range(beta+1):
    for k in range(r+1):
      temp = special.binom(treated_neighb, k) * special.binom(control_neighb, r-k)
      temp1 = ((((1-p)**(k)) * ((-p)**(r-k)))/( p**r) ) - ((((p-1)**(k)) * ((p)**(r-k)))/( (1-p)**r))
      w_vec += temp*temp1

  est = w_vec.dot(y_n) + n*np.mean(outcom)
  if TTE is None:
    return est/n


def unite_estimator_wis_fast(n, p, y, A, z, beta, TTE=None):
  treated_neighb = A.dot(z)
  control_neighb = A.dot(1-z)


  zp = z/p
  zp_ = (1-z)/(1-p)
  szp = A.dot(zp)
  szp_ = A.dot(zp_)
  num_nb_vec = A.dot(np.ones_like(z))

  p1 = p*np.mean(szp/num_nb_vec)
  p1_ = (1-p)*np.mean(szp_/num_nb_vec)

  #print ( "WIS Ps are: ", p ," " , p1 , " " , p1_)

  w_vec = np.zeros(n)
  for r in range(beta+1):
    for k in range(r+1):
      temp = special.binom(treated_neighb, k) * special.binom(control_neighb, r-k)
      temp1 = ((((p1_)**(k)) * ((-p1)**(r-k)))/( p1_**r) ) - ((((-p1_)**(k)) * ((p1)**(r-k)))/( (p1_)**r))
      w_vec += temp*temp1

  est = w_vec.dot(y)
  if TTE is None:
    return est/n



