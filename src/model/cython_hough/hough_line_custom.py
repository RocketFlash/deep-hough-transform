from ._line_hough import _hough_line_custom

def hough_line(image, numangle,
                      numrho,
                      H,
                      W):
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')

    return _hough_line_custom(image, numangle,
                                     numrho,
                                     H,
                                     W)