# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:52:00 2017

@author: C5232886
"""

import tensorflow as tf

av,bv = input().strip().split(' ')
av,bv = [int(av), int(bv)]

#s,n,m = input().strip().split(' ')
#s,n,m = [int(s),int(n),int(m)]

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

print(node1, node2) 
#Does not print the values till a session is established

sess = tf.Session()
print(sess.run([node1, node2]))
#Displays the values in the nodes

node3 = tf.add(node1, node2)
print("node3 value:", node3)
print("sess.run(node3) value:", sess.run(node3))

#A place holder is a promise to provide a value later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, {a:av, b:bv}))
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

add_and_triple = adder_node * 3
print("add_and_triple",sess.run(add_and_triple, {a:av, b:bv}))