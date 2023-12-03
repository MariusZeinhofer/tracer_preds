This is a dataset consisting of 30 pairs of 256x256 PNG images displaying a sagittal MRI slice.

Preprocessing:
With 0h we refer to the baseline image and with 24h we refer to the image taken 24h after tracer injection. Both images
are assumed to be normalized by the signal strength of the fat behind the eye. We are interested in the increase of the signal due to the tracer. 

The following preprocessing has been carried out:

1) compute signal increase ratio (SIR) by the formula SIR = (24h - 0h)/(0h + 0.1).
	the addition of 0.1 prevents division by zero.

2) The target for the learning problem is obtained by cliping SIR, i.e., we do TARGET=min(max(minval, SIR, maxval)
	where we set minval=0, maxval=4. That means we clip tracer injection beyond 400%.

This means that the training pairs are (x,y) = (0h, TARGET) 
