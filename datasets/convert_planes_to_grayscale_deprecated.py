from skimage import io
import matplotlib.pyplot as plt

for i in range(1,800):
    if i%100 == 0:
        print(i)
    path = "equal_scale_planes/" + str(i) + ".jpg"
    img = io.imread(path, as_grey=True)
    # print(img.shape)
    save_path = 'grayscale_planes/' +str(i) +'.jpg'
    io.imsave(save_path, img)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(img, cmap="gray")
    # plt.show()
