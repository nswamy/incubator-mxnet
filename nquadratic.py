import mxnet as mx

x = mx.nd.ones((3,4))

y = mx.nd.nquadratic(x, 2, 3, 4)

print(y)
z1 = mx.sym.Variable("data")
z = mx.sym.nquadratic(z1, a=2, b=3, c=4)

e = z.bind(mx.cpu(), {"data": mx.nd.ones((3,4))})
y1 = e.forward()
print(y1[0].asnumpy())
