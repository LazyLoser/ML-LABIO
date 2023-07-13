import imageio.v2 as imageio
import matplotlib.pyplot as plt
pic = imageio.imread('wallpaper.jpg')
plt.imshow(pic)
print('Type of image',type(pic))
print('Shape of image:{}'.format(pic.shape))
print('image height:{}'.format(pic.shape[0]))
print('image width:{}'.format(pic.shape[1]))
megapixels = (pic.shape[0]*pic.shape[1]/1000000)
print('megapixels:{}'.format(megapixels))
print('dimensions of image:{}'.format(pic.ndim))
print('Value of R channel:{}'.format(pic[100,50,0]))
print('Value of G channel:{}'.format(pic[100,50,1]))
print('Value of B channel:{}'.format(pic[100,50,2]))
