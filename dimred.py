"""Usage: dimred.py tsne <datafilepath> <labelsfilepath>

datafilepath:
    --datafilepath=<str>  read in data from file path

labelsfilepath:    
    --labelsfilepath=<str> read in labels from file path

"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from helperfun import adjustbeta, pca
from docopt import docopt

class TSNE:
    """ TSNE dimensionality reduction

    t-distributed stochastic neighbor embedding (TSNE) nonlinear dimensionality reduction.

    Attributes:
    ----------
        arrNxdInput : `np.ndarray` 
            Input array (n,d).  Should contain numerical data.
        intNDims : `int`, default = 2
            Number of columns in the output array
        dPerplexity : `float`, default = 30.0
            Precision term
        dInitMomentum: `float`, default = 0.5
            Initial momentum of gradient
        dFinalMomentum: `float`, default = 0.8
            Final momentum of gradient
        intInitalMomentumSteps : `int` = 20
            Number of iterations with initial momentum
        dTolerance: `float`, default = 1e-5
            #TODO Figure out what tolerance is
        dMinGain : `float`, default = 0.01
            #TODO Figure out what tolerance is
        intStepSize : `int`, default = 500
            inital step size of gradient descent
        dMinProb : `float`, default = 1e-12
            Clip value for probabilities.  This factor is added to all values of probability array.
        intEarlyExaggerationFactor : `float`, default = 4
            Optimization exaggerates early steps to allow larger movements.
            If `intEarlyExaggerationFactor = 1`, there is no exaggeration
        intNumExaggerationSteps : `int`, default = 100
            Number of steps before exaggeration factor is removed from p_ij
        intMaxIter : `int`, default = 1000
            Number of TSNE iterations
        intUpdateStepSize: `int`, default = 10
            Number of steps before printing update value to console

    Methods
    -------
        SquareEucDist
            Calculate the pairwise squared euclidean distance of input m x n array.
        GetAdjustedBeta
            Calculate the conditional probability matrix from pairwise distances and determine the precision (beta)
        Calc_p_ij
            Probability array representing distance via Gaussian distribution
        Calc_q_ij
            Probability array representing distance via t-distribution
        Calc_dY
            Calculate the gradient of ...
        Calc_gains
            #TODO figure out what gains are ...
        TSNE
            Default implementation of TSNE.  Returns dimension reduced array.
    """

    def __init__(
        self,
        arrNxdInput: np.ndarray,
        intNDims: int = 2,
        intMaxIter: int = 1000,
        dPerplexity: float = 30,
        dInitMomentum: float = 0.5,
        dFinalMomentum: float = 0.8,
        intInitalMomentumSteps: int = 20,
        dTolerance: float = 1e-5,
        intStepSize: int = 500,
        dMinGain: float = 0.01,
        dMinProb: float = 1e-12,
        dEarlyExaggerationFactor: float = 4,
        intNumExaggerationSteps: int = 100,
        intUpdateStepSize: int = 10,
    ):
        # Initalize all variables
        self.arrNxdInput = arrNxdInput
        self.intNDims = intNDims
        self.dPerplexity = dPerplexity
        self.dInitMomentum = dInitMomentum
        self.dFinalMomentum = dFinalMomentum
        self.dTolerance = dTolerance
        self.dMinGain = dMinGain
        self.dMinProb = dMinProb
        self.dEarlyExaggerationFactor = dEarlyExaggerationFactor
        self.intNumExaggerationSteps = intNumExaggerationSteps
        self.intInitalMomentumSteps = intInitalMomentumSteps
        self.intUpdateStepSize = intUpdateStepSize
        self.intStepSize = intStepSize
        self.intMaxIter = intMaxIter
        self.arr1DGains = np.ones((arrNxdInput.shape[0], intNDims))
        self.arr1DDeltaY = np.zeros((arrNxdInput.shape[0], intNDims))
        self.arrNxNSquareEucDist = self.SquareEucDist(arrNxdInput)
        self.arrNxNDimsOutput = arrNxdInput[:, :intNDims]
        self.arr1DBeta = None
        self.arrNxNInputProbs = None
        self.KLDiv = None

    def SquareEucDist(self, array_nxd: np.ndarray) -> np.ndarray:
        """Pairwise Squared Euclidian distance
        .. math::
            \sum_{i \neq j}||x_{i} - x_{j}||^{2}
        
        Parameters
        ----------
        array_nxd : np.ndarray
            Input array (n,d) 

        Returns 
        -------
        arrNxNSquareEucDistMat : np.ndarray
            Pairwise distance array (n,n)
        """
        Xsum = np.sum(np.square(array_nxd), axis=1)
        twoXiXj = 2 * np.dot(array_nxd, array_nxd.T)
        arrNxNSquareEucDistMat = Xsum[:, None] - twoXiXj + Xsum
        return arrNxNSquareEucDistMat

    def GetAdjustedBeta(self):
        """ Get the beta using tolerance and perplexity

        Returns
        -------
        arrInputProbs : np.ndarray
            probability matrix (n,n)
        arr1DBeta : np.ndarray
            precision array (n, 1)

        Notes
        -----
        This is a wrapper for the adjustbeta.adjustbeta function.

        See Also
        --------
        adjustbeta.py
        """
        self.arrInputProbs, self.arr1DBeta = adjustbeta(
            self.arrNxdInput,
            self.dTolerance,
            self.dPerplexity,
            self.arrNxNSquareEucDist,
        )
        return self.arrInputProbs, self.arr1DBeta

    def Calc_p_ij(self) -> np.ndarray:
        """ Gaussian probability matrix from input array pairwise distances
        .. math::
            p_{ij} = \frac{p_{j|i} + p_{i|j}}/{\sum_{k \neq l}p_{k|l} + p_{l|k}}
            where:
            p_{j|i} = \frac{exp(-||x_{i} - x_{j}||^{2}*\beta)}/{\sum_{i \neq k}exp(-||x_{i} - x_{k}||^{2}*\beta)}
            p_{i|j} = p_{j|i}.T
            \beta = \frac{1}/{(2 * \sigma^2)}

        Returns
        -------
        arrInputProbs_ij_mat : np.ndarray
            Gaussian representation fo pairwise distance probabilies 

        Notes
        -----
        By default the returned prob array is clipped to `self.dMinProb` and exaggerated by `self.dEarlyExaggerationFactor`
        """
        # Set diag to 0.0
        np.fill_diagonal(self.arrInputProbs, 0.0)
        # Calc Pij from Probability matrix P
        num = self.arrInputProbs + self.arrInputProbs.T
        denom = np.sum(num)
        # Num / denom; exaggerate (*4) and clip (1e-12)
        self.arrInputProbs_ij_mat = (
            self.dEarlyExaggerationFactor * (num / denom) + self.dMinProb
        )
        return self.arrInputProbs_ij_mat

    def Calc_q_ij(self) -> np.ndarray:
        """ Represent distance with t-distribution (df = 1)
        .. math::
            q_{ij} = \frac{1 + -||y_{i} - y_{j}||^{2}}^{-1}/{\sum_{i \neq k}1 + -||y_{k} - y_{l}||^{2}}^{-1}

        Returns
        -------
        q_ij_mat : np.ndarray
            The t-distribution representation of pairwise distances
        """
        Q = self.SquareEucDist(self.arrNxNDimsOutput)
        num = (1 + Q) ** -1
        # Set diag to 0.0
        np.fill_diagonal(num, 0.0)
        denom = np.sum(num)
        # num/denom then clip
        self.q_ij_mat = (num / denom) + self.dMinProb
        return self.q_ij_mat

    def Calc_dY(self) -> np.ndarray:
        """ Calculate the gradient for Y"""
        self.dY = np.sum(
            (self.arrInputProbs_ij_mat - self.q_ij_mat)[:, :, None]
            * (self.arrNxNDimsOutput[:, None] - self.arrNxNDimsOutput[None, :])
            * ((1 + self.SquareEucDist(self.arrNxNDimsOutput)) ** -1)[:, :, None],
            axis=1,
        )
        return self.dY

    def Calc_gains(self) -> np.ndarray:
        """ Calculate the gain form signs of dY and deltaY
        """
        self.arr1DGains = (self.arr1DGains + 0.2) * (
            (self.dY > 0.0) != (self.arr1DDeltaY > 0.0)
        ) + (self.arr1DGains * 0.8) * ((self.dY > 0.0) == (self.arr1DDeltaY > 0.0))
        return self.arr1DGains

    def KLDivergence(self):
        """Calculate the KL Divergence for the current step"""
        self.KLDiv = np.sum(self.arrInputProbs_ij_mat * np.log(self.arrInputProbs_ij_mat / self.q_ij_mat))
        return self.KLDiv

    def TSNE(self) -> np.ndarray:
        """TSNE
        
        Returns
        -------
        arrNxNDimsOutput : np.ndarray
            The dimensionally reduced output array
        """
        self.SquareEucDist(self.arrNxdInput)
        self.GetAdjustedBeta()
        self.Calc_p_ij()

        for i in range(self.intMaxIter):
            self.Calc_q_ij()
            self.Calc_dY()

            if i < self.intInitalMomentumSteps:
                dCurrMomentum = self.dInitMomentum
            else:
                dCurrMomentum = self.dFinalMomentum

            self.arr1DGains = self.Calc_gains() + self.dMinGain
            self.arr1DDeltaY = dCurrMomentum * self.arr1DDeltaY - self.intStepSize * (
                self.arr1DGains * self.dY
            )
            self.arrNxNDimsOutput += self.arr1DDeltaY

            if i == self.intNumExaggerationSteps:
                # Remove early exaggeration
                self.arrInputProbs_ij_mat = (
                    self.arrInputProbs_ij_mat / self.dEarlyExaggerationFactor
                )

            if i % self.intUpdateStepSize == 0:
                KLDiv = self.KLDivergence()
                print(f'Step: {i}     KLDiv: {KLDiv}')
        return self.arrNxNDimsOutput

def main():
    args = docopt(__doc__)
    
    print(f"Loading data from file {args['<datafilepath>']}...")
    X = np.loadtxt(args['<datafilepath>'])
    print('load successful.')
    print('performing pca...')
    X = pca(X, 50)
    print('pca complete...')

    labels = np.loadtxt(args['<labelsfilepath>'])
    print('labels loaded successfully...')
    
    print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    tsne = TSNE(X, intMaxIter=100)
    Y = tsne.TSNE()

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.savefig("mnist_tsne.png")

if __name__ == "__main__":
    main()