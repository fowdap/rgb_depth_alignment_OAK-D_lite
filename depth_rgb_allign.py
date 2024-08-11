import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()
fps = 30
rgbWeight = 0.4
depthWeight = 0.6

def updateBlendWeights(percent_rgb):
    """
    Update the rgbWeight and depthWeight to blend
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight

# Nodes creation for the pipeline

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
colorCamera = pipeline.create(dai.node.ColorCamera)

# output nodes
xoutDepth = pipeline.create(dai.node.XLinkOut)
colorOut = pipeline.create(dai.node.XLinkOut)

# stream names
xoutDepth.setStreamName("depth")
colorOut.setStreamName("rgb")

# properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setCamera("left")
monoLeft.setFps(fps)

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")
monoRight.setFps(fps)

colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCamera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
colorCamera.setFps(fps)


stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)


colorCamera.isp.link(colorOut.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

with dai.Device(pipeline) as device:
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lensPosition:
            colorCamera.initialControl.setManualFocus(lensPosition)
    except:
        raise

    
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    colorQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    blendedWindowName = "rgb-depth"
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)


    while True:
        colorFrame = colorQueue.get().getCvFrame()
        depthFrame = depthQueue.get().getFrame()

        if colorFrame is not None:
            # print(depthFrame.shape)
            cv2.imshow("colorFrame", colorFrame)

        if depthFrame is not None:
            # print(colorFrame.shape)
            depthFrame = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrame = cv2.equalizeHist(depthFrame)
            depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_HOT)
            # print(len(depthFrame.shape))
            cv2.imshow("depth-frame", depthFrame)

        if colorFrame is not None and depthFrame is not None:
            if len(depthFrame.shape) < 3:
                depthFrame = cv2.cvtColor(depthFrame, cv2.COLOR_GRAY2BGR)
            blendedFrame = cv2.addWeighted(colorFrame, rgbWeight, depthFrame, depthWeight, 0)
            cv2.imshow(blendedWindowName, blendedFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

