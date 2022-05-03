""" Start of file doc string


# TODO: Get KL Divergence implemented
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from tsne import pca
from adjustbeta import Hbeta, adjustbeta


class TSNE:
    """ TSNE dimensionality reduction

    Attributes:
    ----------
        arrNxdInput : np.array 
        intNDims : int, default = 2
        dPerplexity : float, default = 30.0
        dInitMomentum: float, default = 0.5
        dFinalMomentum: float, default = 0.8
        dTolerance: float, default = 1e-5
        dMinGain : float, default = 0.01
        intStepSize: int, default = 500
        intMaxIter: int, default = 1000

    Methods
    -------
        SquareEucDist
            Calculate the pairwise squared euclidean distance of input m x n array.
        GetAdjustedBeta
            d
        Calc_p_ij
            d
        Calc_q_ij
            d
        Calc_dY
            d
        Calc_gains
            d
        TSNE
            d
    """

    def __init__(
        self,
        arrNxdInput: np.array,
        intNDims: int = 2,
        dPerplexity: float = 30,
        dInitMomentum: float = 0.5,
        dFinalMomentum: float = 0.8,
        dTolerance: float = 1e-5,
        intStepSize: int = 500,
        dMinGain: float = 0.01,
        intMaxIter: int = 250,
    ):
        """Initialize all variables"""
        self.arrNxdInput = arrNxdInput
        self.intNDims = intNDims
        self.dPerplexity = dPerplexity
        self.dInitMomentum = dInitMomentum
        self.dFinalMomentum = dFinalMomentum
        self.dTolerance = dTolerance
        self.intStepSize = intStepSize
        self.dMinGain = dMinGain
        self.intMaxIter = intMaxIter
        self.arr1DGains = np.ones((arrNxdInput.shape[0], intNDims))
        self.arr1DDeltaY = np.zeros((arrNxdInput.shape[0], intNDims))
        self.arrNxNSquareEucDist = self.SquareEucDist(arrNxdInput)
        self.arrNxNDimsOutput = arrNxdInput[:, 0:2]
        self.arr1DBeta = None
        self.arrNxNInputProbs = None
        self.H = None

    def SquareEucDist(self, array_nxm: np.array):
        """Pairwise Squared Euclidian distance



        """
        Xsum = np.sum(np.square(array_nxm), axis=1)
        twoXiXj = 2 * np.dot(array_nxm, array_nxm.T)
        SquareEucDistMat = Xsum[:, None] - twoXiXj + Xsum
        return SquareEucDistMat

    def GetAdjustedBeta(self):
        """ Get the beta using tolerance and perplexity
        """
        self.arrInputProbs, self.beta = adjustbeta(
            self.arrNxdInput,
            self.dTolerance,
            self.dPerplexity,
            self.arrNxNSquareEucDist,
        )
        return self.arrInputProbs, self.beta

    def Calc_p_ij(self):
        """ Probability matrix
        """
        # Set diag to 0.0
        np.fill_diagonal(self.arrInputProbs, 0.0)
        # Calc Pij from Probability matrix P
        num = self.arrInputProbs + self.arrInputProbs.T
        denom = np.sum(np.sum(num, axis=1), axis=0)
        # Num / denom; exaggerate (*4) and clip (1e-12)
        self.arrInputProbs_ij_mat = 4 * (num / denom) + 1e-12
        return self.arrInputProbs_ij_mat

    def Calc_q_ij(self):
        """
        """
        Q = self.SquareEucDist(self.arrNxNDimsOutput)
        num = (1 + Q) ** -1
        # Set diag to 0.0
        np.fill_diagonal(num, 0.0)
        denom = np.sum(num)
        # num/denom then clip
        self.q_ij_mat = (num / denom) + 1e-12
        return self.q_ij_mat

    def Calc_dY(self):
        """ Calculate the gradient for Y"""
        self.dY = np.sum(
            (self.arrInputProbs_ij_mat - self.q_ij_mat)[:, :, None]
            * (self.arrNxNDimsOutput[:, None] - self.arrNxNDimsOutput[None, :])
            * ((1 + self.SquareEucDist(self.arrNxNDimsOutput)) ** -1)[:, :, None],
            axis=1,
        )
        return self.dY

    def Calc_gains(self):
        """ Calculate the gain form signs of dY and deltaY
        """
        self.arr1DGains = (self.arr1DGains + 0.2) * (
            (self.dY > 0.0) != (self.arr1DDeltaY > 0.0)
        ) + (self.arr1DGains * 0.8) * ((self.dY > 0.0) == (self.arr1DDeltaY > 0.0))
        return self.arr1DGains

    def TSNE(self):
        """Do the TSNE"""
        self.SquareEucDist(self.arrNxdInput)
        self.GetAdjustedBeta()
        self.Calc_p_ij()

        for i in range(self.intMaxIter):
            self.Calc_q_ij()
            self.Calc_dY()

            if i < 20:
                dCurrMomentum = self.dInitMomentum
            else:
                dCurrMomentum = self.dFinalMomentum

            self.arr1DGains = self.Calc_gains() + self.dMinGain
            self.arr1DDeltaY = dCurrMomentum * self.arr1DDeltaY - self.intStepSize * (
                self.arr1DGains * self.dY
            )
            self.arrNxNDimsOutput += self.arr1DDeltaY

            if i == 100:
                # Remove early exaggeration
                self.arrInputProbs_ij_mat = self.arrInputProbs_ij_mat / 4

            if i % 10 == 0:
                print(i)
        return self.arrNxNDimsOutput


if __name__ == "__main__":
    print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    labels = np.loadtxt("mnist2500_labels.txt")
    tsne = TSNE(X)
    Y = tsne.TSNE()

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.savefig("mnist_tsne.png")
