# Python Implementation of Algorithm for Estimating the Unseen Entropy
Project Title: Non-matlab implementation of algorithm described in [“Estimating the Unseen: Improved Estimators for Entropy and other Properties”](https://theory.stanford.edu/~valiant/papers/unseenJournal.pdf)
Course Code: CSE 549.01
Institution : Stony Brook University
Team Members: Neel Paratkar(111483570), Rohan Karhadkar (), Sharad Sridhar (), Chaitanya Kalantri()

## Paper Abstract
There  are  many  methods  to  find  the  relationship  among  the  data.  Forinstance,  when  finding  the  distribution  of  random  sample,  empirical methods may not always be efficient.  As the distribution can be linear,non-linear or may not even be distributed in any definable pattern.And in real world datasets. The data is highly random and can’t alwaysbe categorized into uniform distributions. Hence, in the paper, we introducea different entropy function, which is used to find the relationship amongthe data. It is more robust to find the "unseen" patterns within the data.We can estimate the shape or the histogram of the unseen portion of the data. And now, given such a reconstruction, one can any property of thedistribution which only depends on the shape/histogram; such propertiesare termed symmetric and include entropy and support size. In   the   paper,    the   algorithm   proposed   is   based   on   the   linearprogramming.  It  does  not  even  have  an  objective  function  and  simplydefines a feasible region.

## System Requirements
* [Python 2/3](https://www.python.org/downloads/)
* [SciPy Library](https://www.scipy.org/)
* [NumPy Library](https://www.scipy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org/)

## Code Details:
* **unseen(f, gridFactor=1.05) ; returns histx, x**
* **entropy_entC(f, gridFactor=1.05); retuns est_entropy**
