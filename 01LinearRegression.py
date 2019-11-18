import numpy as np

# 计算Loss
def cpmpute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        totalError += (y - (w * x + b)) ** 2

    return totalError / float(len(points))


# 计算梯度
def step_grad(b_current, w_current, points, lr):
    b_grad = 0
    w_grad = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_grad += -(2 / N) * (y - ((w_current * x) + b_current))
        w_grad += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (lr * b_grad)
    new_w = w_current - (lr * w_grad)
    return [new_b, new_w]


# 迭代优化器
def gra_des_runner(points, starting_b, starting_w, l_r, num_iter):
    b = starting_b
    w = starting_w
    for i in range(num_iter):
        b, w = step_grad(b, w, np.array(points), l_r)
    return [b, w]


def run():
    points = np.genfromtxt("Linear_data.csv", delimiter=',')
    lr = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 100
    print("Starting gradient descent at b = {0},w = {0},error = {2}"
          .fromat(initial_b, initial_w, points)
          )
    print("Running....")

    [b, w] = gra_des_runner(points, initial_b, initial_w,lr, initial_w)
    print("After {0} iteration b = {1}, w = {2},error = {3}".
          format(num_iterations, b, w,
                 cpmpute_error_for_line_given_points(b, w, points)
                 )
          )


if __name__ == ' main ':
    run()