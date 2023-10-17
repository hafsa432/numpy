#!/usr/bin/env python
# coding: utf-8

# # Name :Hafsa Rani
# 
# 

# # Reg no # 0252

# # Decomentation of Numpy

# # What is NumPy?
# 
# 

# ➤NumPy is the fundamental package for scientific computing in Python.
# 
# ➤ NumPy is a Python library that provides a multidimensional array object, various derived objects

# ## Installing NumPy

# Before you start using NumPy, you need to make sure it's installed on your system. 
# You can typically install it using pip, the Python package manager:

# pip install numpy

# ## How to import NumPy
# 

# To get to NumPy and its capabilities import it in your Python code like this:

# import numpy as np

# ## Why use NumPy?

# ➤NumPy provides efficient storage
# 
# ➤It also provides better ways of handling data for processing
# 
# ➤It is fast
# 
# ➤It is easy to learn
# 
# ➤NumPy uses relatively less memory to store data

# # What is NumPy Array ?

# An array is a grid of values and it contains information about the raw data, how to locate an element, and how to interpret an element.

# In[1]:


import numpy as np
y= np.array([1,2,3,4,])
print(y)
print(type(y))


# # Dimensions in Arrys

# 1-D array -----> [1 2 3 4 5]
# 

# 2-D array -----> [[1 2 3 4 5 ]]
# 

# 3-D array -----> [[[1 2 3 4 5]]]

#  You can find the dimension through this function  (ndim) 

# # 1-D Array

# In[2]:


import numpy as np
y= np.array([1,2,3,4,])
print(y)
print(type(y))
print(y.ndim)


# # 2-D Array

# In[3]:


ar2 = np.array([[1,2,3,4,5],[1,2,3,4,5]])
print(ar2)
print(ar2.ndim)


# # 3-D Array

# In[4]:


ar3 = np.array([[[1,2,3,4,5],[1,9,3,8,5],[1,2,6,7,4,]]])
print(ar3)
print(ar3.ndim)


# # Multi D- Array

# In[5]:


arn = np.array([1,2,3,4,5], ndmin=10)
print(arn)
print(arn.ndim)


# # **Special numpy array**

# # Array filled with 0's
# 

# In[6]:


import numpy as np
ar_zero = np.zeros(4)
print(ar_zero)


# In[7]:


import numpy as np
ar_zero1 = np.zeros((3,4))
print(ar_zero1)


# # Array filled with 1's

# In[8]:


ar_ones =np.ones(4)
print(ar_ones)


# # Create an empty array

# In[9]:


ar_em = np.empty(4)
print(ar_em)


# # An array with a range of elements

# In[10]:


ar_rn = np.arange(4)
print(ar_rn)


# # Array diagonal element filled with 1's

# In[11]:


ar_dia = np.eye(3)
print(ar_dia)


# # Create an array with valuethat are spaced linearly in a specified interval

# # linspace

# In[12]:


ar_lin = np.linspace(0,20,num=5)
print(ar_lin)


# # Random

# # rand() the function is used to generatea random value b/w  0 to1

# In[13]:


import numpy as np
var = np.random.rand(4)
print(var)                                     #1-D ARRAY


# In[14]:


import numpy as np
var1 = np.random.rand(3,5)
print(var1)                                   #2-D ARRAY


# # randn()  the function is used to generatea random value close to zero. this may return positive or negative number as well

# In[15]:


var2 = np.random.randn(5)

print(var2)


# # ranf() the function for doing randomsampling in numpy.it returns an array of specified shape and fills it with random floats in the half open interval[0.0,1.0)

# In[16]:


var3 = np.random.ranf(4)
print(var3)


# # randint() the function is used to generate a random number b/w a given range.

# In[17]:


var4 = np.random.randint(5,20,5)   

print(var4) 


# # Data Type

# In[18]:


import numpy as np
var = np.array([1,2,4,6,11,14,15])
print("Data Type : ",var.dtype)


# In[19]:


var = np.array([1,0.1,0.5,8,9])
print("Data Type : ",var.dtype)


# In[20]:


var = np.array(["A","W","Y"])
print("Data Type : ",var.dtype)


# In[21]:


import numpy as np
x = np.array([1,2,3,4,],dtype = np.int8)
print("Data Type : ",x.dtype)
print(x)


# In[22]:


import numpy as np
x1 = np.array([1,2,3,4,],dtype = 'f')
print("Data Type : ",x1.dtype)
print(x1)


# # Ariathmetic operation in Numpy array

# # ➤a+b ----------->np.add(a,b)
# 
# 

# In[23]:


import numpy as np
var = np.array([1,2,3,4,5])
varadd = var+3
print(varadd)


# In[24]:


var1 = np.array([1,6,3,4,7])
var2 = np.array([1,6,3,4,7])
varadd = var1+var2
print(varadd)


# In[25]:


var1 = np.array([1,6,3,4,7])
var2 = np.array([1,6,3,4,7])
varadd = np.add(var1,var2)   # use function
print(varadd)


# # ➤a-b----------->np.subtract(a,b)
# 

# In[26]:


var = np.array([1,6,3,4,7])
varadd = var-3
print(varadd)


# # ➤a*b---------->np.multiply(a,b)
# 

# In[27]:


var = np.array([1,6,3,4,7])
varadd = var*3
print(varadd)


# # ➤a/b----------->np.divided(a,b)
# 

# In[28]:


var = np.array([1,6,3,4,7])
varadd = var/3
print(varadd)


# # ➤a%b---------->np.mod(a,b)
# 

# In[29]:


var = np.array([1,6,3,4,7])
varadd = var%3
print(varadd)


# # ➤a**b----------->np.power(a,b)
# 

# In[30]:


var = np.array([1,6,3,4,7])
varadd = var**2
print(varadd)


# # ➤1/a---------->np.reciprocal(a)

# In[31]:


var = np.array([1,6,3,4,7])
varadd = np.reciprocal(var)
print(varadd)


# # ➤np.min(x)

# # ➤np.max(x)

# In[32]:


import numpy as np
var = np.array([1,2,3,4,2,4,2])
print("min :",np.min(var))
print("max :",np.max(var))


# # ➤np.argmin(x)

# In[33]:


var = np.array([1,2,3,4,2,4,2])
print("min :",np.min(var),np.argmin(var))
print("max :",np.max(var),np.argmax(var))        #fun to find indexes no


# # ➤np.sqrt(x)

# In[34]:


var = np.array([1,2,3,4])
print("sqrt : ",np.sqrt(var))


# # ➤np.sin(x)
# 

# In[35]:


var = np.array([1,2,3,])
print(np.sin(var))


# # ➤np.cos(x)
# 

# In[36]:


var = np.array([1,2,3,])
print(np.cos(var))


# # ➤np.cumsum(x)

# In[37]:


var = np.array([1,2,3,4])
print(np.cumsum(var))


# # shape and reshaping in numpy array

# # shape

# In[38]:


import numpy as np
var = np.array([[1,2,3,4,],[1,2,3,4]])
print(var)
print()
print(var.shape)


# In[39]:


var1 = np.array([1,2,3,4,],ndmin=4) #multi D array
print(var1)
print(var1.ndim)
print()
print(var1.shape)


# # reshape

# In[40]:


var2 = np.array([1,2,3,4,5,6])
print(var2)
print(var2.ndim)        
print()                 
x =var2.reshape(3,2)
print(x)
print(x.ndim)


# In[41]:


var3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,])
print(var3)
print(var3.ndim)        
print()                 
x1 =var3.reshape(2,3,2)
print(x1)
print(x1.ndim)


# In[42]:


var3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,])
print(var3)
print(var3.ndim)        
print()                 
x1 =var3.reshape(2,3,2)
print(x1)
print(x1.ndim)

one = x1.reshape(-1)         #reshape
print(one)
print(one.ndim)


# # Broadcasting In Numpy Arrays

# 1)same dimension

# In[53]:


import numpy as np
var1 = np.array([1,2,3])
print(var1.shape)
print(var1)
print()

var2 = np.array([[1],[2],[3]])
print(var2.shape)

print()
print(var2)
print()
print(var1+var2)


# In[58]:


x= np.array([[1],[2]])
print (x.shape)

y=np.array([[1,2,3],[1,2,3]])
print(y.shape)
print (x+y)


# # Indexing and Slicing In NumPy Arrays

# # indexing

# In[60]:


import numpy as np
var = np.array([9,8,7,6])
#               0 1 2 3
#              -4,-3,-2,-1
print(var[1])
print(var[-3])


# In[63]:


var = np.array([[9,8,7],[4,5,6]])                 #2-D array
print(var)
print(var.ndim)
print()

print(var[0,2])


# In[67]:


var1 = np.array([[[1,2],[1,4]]])
print(var1)
print(var1.ndim)


# # slicing

# In[76]:


import numpy as np
var = np.array([1,2,3,4,5,6,7,8,9])
#               0,1,2,3,4,5,6,7,8
print(var)

print()
print("2 to 5 : ",var[1:5])
print("2 to end : ",var[1:])        #end point
print("start to 5 : ",var[:5])     #start point
print("stop to 6: ",var[1:5:2])         #stop point


# In[ ]:




