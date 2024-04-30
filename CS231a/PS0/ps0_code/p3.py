# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def low_rank_approx(rank, u, s, v):
    pass # we recommend you implement this helper function for parts b and c

def part_a():
    # ===== Problem 4a =====
    # Take the singular value decomposition of the image.

    img1 = io.imread('image1.jpg', as_gray=True)
    u, s, v = None, None, None

    # BEGIN YOUR CODE HERE
    u, s, v = np.linalg.svd(img1)
    # END YOUR CODE HERE
    
    return u, s, v

def part_b(u, s, v):
    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.
    rank1approx = None

    # BEGIN YOUR CODE HERE
    rank1_s = np.zeros_like(s)
    rank1_s[:1] = s[:1]
    rank1approx = np.dot(u, np.diag(rank1_s)).dot(v)
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(rank1approx, cmap = plt.get_cmap('gray'))
    plt.savefig('p3b.png')
    return rank1approx
    
def part_c(u, s, v):
    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.
    rank20approx = None

    # BEGIN YOUR CODE HERE
    rank20_s = np.zeros_like(s)
    rank20_s[:20] = s[:20]
    rank20approx = np.dot(u, np.diag(rank20_s)).dot(v)
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(rank20approx, cmap = plt.get_cmap('gray'))
    plt.savefig('p3c.png')
    return rank20approx

if __name__ == '__main__':
    u, s, v = part_a()
    rank1approx = part_b(u, s, v)
    rank20approx = part_c(u, s, v)