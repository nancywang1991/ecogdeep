import matplotlib.pyplot as plt
import numpy as np
import glob
end = int((3500 + 999) * (30 / 1000.0))

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

mvmt = np.zeros(shape=(56,56))
for file in glob.glob("C:\\Users\\Nancy\\Downloads\\Y\\*.npy"):
    print file
    ydata = np.load(file)[(end - 15):end]
    ydata_start = ydata[0]
    ydata_end = ydata[-1]
    t = 0
    while ydata_start[0] < 0:
        t += 1
        ydata_start = ydata[t]
    t = -1
    while ydata_end[0] < 0:
        t -= 1
        ydata_end = ydata[t]
    ydata_end[0] = ydata_end[0] * (256 / 640.0)
    ydata_end[1] = ydata_end[1] * (256 / 480.0)
    mvmt += makeGaussian(56, center=(ydata_end) / 4.0)
plt.imshow(mvmt)
plt.show()
