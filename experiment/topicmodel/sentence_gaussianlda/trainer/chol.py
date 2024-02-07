"""
Implementation of the collapsed gibbs sampler
for LDA where the distribution of each table are multivariate gaussian
with unknown mean and covariances.

Closely based on the authors' implementation in Java:
  https://github.com/rajarshd/Gaussian_LDA

This implementation uses Numpy/Scipy.

This is a basic implementation that implements the simple version of the training with no
speed-up tricks, and also the Cholesky decomposition, which can be disabled.

"""
import json
import math
import os
import pickle
import shutil

import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import inv, solve_triangular
from scipy.special import gammaln
import torch

from sentence_gaussianlda.src.prior import Wishart
from sentence_gaussianlda.src.utils import get_logger, get_progress_bar, chol_rank1_downdate, chol_rank1_update
from sentence_gaussianlda.src.perplexity import calculate_avg_ll


class GaussianLDATrainer:
    def __init__(self, corpus, encoder, num_tables, alpha, kappa, log=None, save_path=None,
                 cholesky_decomp=True, num_words_for_formatting=None):
        """
        :param corpus:
        :param vocab_embeddings:
        :param vocab:
        :param num_tables:
        :param alpha:
        :param log:
        :param save_path:
        :param show_topics:
        :param cholesky_decomp:
        :param num_words_for_formatting: By default, each topic is formatted by computing the probability of
            every word in the vocabulary under that topic. This can take a long time for a large vocabulary.
            If given, this limits the number considered to the first
            N in the vocabulary (which makes sense if the vocabulary is ordered with most common words first).
        """
        if log is None:
            log = get_logger("GLDA")
        self.log = log
        # Vocab is used for outputting topics
        self.cholesky_decomp = cholesky_decomp
        self.save_path = save_path

        # Dirichlet hyperparam
        self.alpha = alpha

        # dataVectors
        self.encoder = encoder
        self.embedding_size = encoder.get_sentence_embedding_dimension()
        # List of list of ints
        self.corpus = corpus
        # numIterations
        # K, num tables
        self.num_tables = num_tables
        # N, num docs
        self.num_documents = len(corpus)
        # In the current iteration, map of table_id's to number of customers. Table id starts from 0
        self.table_counts = np.zeros((self.num_tables), dtype=np.int32)
        # K x N array.tableCounts[i][j] represents how many words of document j are present in topic i.
        self.table_counts_per_doc = np.zeros((self.num_tables, self.num_documents), dtype=np.int32)
        # Stores the table (topic) assignment of each customer in each iteration
        # tableAssignments[i][j] gives the table assignment of customer j of the ith document.
        self.table_assignments = []
        self.average_ll = 0
        # The following 4 parameters are arraylist (list) and not maps (dict) because,
        # if they are K tables, they are continuously numbered from 0 to K-1 and hence we can directly index them
        # Mean vector associated with each table in the current iteration.
        # This is the bayesian mean (i.e has the prior part too)
        self.table_means = np.zeros((self.num_tables, self.embedding_size), dtype=np.float64)
        # inverse of covariance matrix associated with each table in the current iteration.
        # The covariance matrix is scaled before taking the inverse by \frac{k_N + 1}{k_N (v_N - D + 1)}
        # (This is because the t-distribution take the matrix scaled as input)
        self.table_inverse_covariances = np.zeros((self.num_tables, self.embedding_size, self.embedding_size),
                                                  dtype=np.float64)
        # log-determinant of covariance matrix for each table.
        # Since 0.5 * logDet is required in (see logMultivariateTDensity), therefore that value is kept.
        self.log_determinants = np.zeros(self.num_tables, dtype=np.float64)
        # Stores the sum of the vectors of customers at a given table
        self.sum_table_customers = np.zeros((self.num_tables, self.embedding_size), dtype=np.float64)
        # Stores the squared sum of the vectors of customers at a given table
        self.sum_squared_table_customers = np.zeros((self.num_tables, self.embedding_size, self.embedding_size), dtype=np.float64)

        if self.cholesky_decomp:
            # Cholesky Lower Triangular Decomposition of covariance matrix associated with each table.
            self.table_cholesky_ltriangular_mat = np.zeros(
                (self.num_tables, self.embedding_size, self.embedding_size), dtype=np.float64)
        else:
            self.table_cholesky_ltriangular_mat = None

        # Normal inverse wishart prior
        mu = np.mean([self.encoder.encode(i).mean(axis=0) for i in corpus], axis=0)
        self.prior = Wishart(mu, self.embedding_size, kappa=kappa)

        # Cache k_0\mu_0\mu_0^T, only compute it once
        # Used in calculate_table_params()
        self.k0mu0mu0T = self.prior.kappa * np.outer(self.prior.mu, self.prior.mu)

        self.num_words_for_formatting = num_words_for_formatting
        self.average_ll = list()

        self.log.info("Initializing assignments")
        self.initialize()

    def initialize(self):
        """
        Initialize the gibbs sampler state.

        I start with log N tables and randomly initialize customers to those tables.

        """
        # First check the prior degrees of freedom.
        # It has to be >= num_dimension
        if self.prior.nu < self.embedding_size:
            self.log.warn("The initial degrees of freedom of the prior is less than the dimension!. "
                          "Setting it to the number of dimensions: {}".format(self.embedding_size))
            self.prior.nu = self.embedding_size

        deg_of_freedom = self.prior.nu - self.embedding_size + 1
        # Now calculate the covariance matrix of the multivariate T-distribution
        coeff = (self.prior.kappa + 1.) / (self.prior.kappa * deg_of_freedom)
        sigma_T = self.prior.sigma * coeff
        # This features in the original code, but doesn't get used
        # Or is it just to check that the invert doesn't fail?
        #sigma_Tinv = inv(sigma_T)
        sigma_TDet_sign, sigma_TDet = slogdet(sigma_T)
        if sigma_TDet_sign != 1:
            raise ValueError("sign of log determinant of initial sigma is {}".format(sigma_TDet_sign))

        # Storing zeros in sumTableCustomers and later will keep on adding each customer.
        self.sum_table_customers[:] = 0
        self.sum_squared_table_customers[:] = 0
        # With Cholesky: Means are set to the prior and then updated as we add each assignment
        # Without: Means are computed fully for each table after initialization
        self.table_means[:] = self.prior.mu
        # With Cholesky: This is ignored - we never use table_inverse_covariances
        # Without: This gets computed after initialization
        self.table_inverse_covariances[:] = 0

        # Initialize the cholesky decomp of each table, with no counts yet
        if self.cholesky_decomp:
            for table in range(self.num_tables):
                self.table_cholesky_ltriangular_mat[table] = self.prior.chol_sigma.copy()

        # Randomly assign customers to tables
        self.table_assignments = []
        pbar = get_progress_bar(len(self.corpus), title="Initializing")
        for doc_num, doc in enumerate(pbar(self.corpus)):
            tables = list(np.random.randint(self.num_tables, size=len(doc)))
            self.table_assignments.append(tables)
            sentence_encodings = self.encoder.encode(doc)
            for (encoding, table) in zip(sentence_encodings, tables):
                self.table_counts[table] += 1
                self.table_counts_per_doc[table, doc_num] += 1
                # update the sumTableCustomers
                self.sum_table_customers[table] += encoding
                self.sum_squared_table_customers[table] += np.outer(encoding, encoding)

                if self.cholesky_decomp:
                    self.update_table_params_chol(table, encoding)

            #self.check_everything(-1, doc_num, -1)

        # Now compute the table parameters of each table
        # Go over each table.
        if not self.cholesky_decomp:
            for table in range(self.num_tables):
                self.set_table_parameters(table)
        else:
            # Make sure we don't accidentally use the inverse covariances anywhere, as they don't
            # get updated
            self.table_inverse_covariances = None

        self.log.info("Computing average LL")
        ave_ll = calculate_avg_ll(
            get_progress_bar(len(self.corpus))(self.corpus), self.table_assignments, self.encoder,
            self.table_means, self.table_cholesky_ltriangular_mat,
            self.prior, self.table_counts_per_doc
        )
        self.log.info("Average LL after initialization: {:.3e}".format(ave_ll))

        #self.check_everything()

    def calculate_table_params(self, table_id):
        """
        This method calculates the table params (bayesian mean, covariance^-1, determinant etc.)

        All it needs is the table_id and the tableCounts, tableMembers and sumTableCustomers
        should be updated correctly before calling this.

        It's used by set_table_params(), but is separated so you can calculate things without
        updating the stored values. (We don't actually do this anywhere here at the moment.)

        """
        # Total global assignments to table
        count = self.table_counts[table_id]
        k_n = self.prior.kappa + count
        nu_n = self.prior.nu + count

        # Update table mean
        mu_n = (self.sum_table_customers[table_id] + self.prior.mu * self.prior.kappa) / k_n

        # we will be using the new update
        # Sigma_N = Sigma_0 + \sum(y_iy_i^T) - (k_n)\mu_N\mu_N^T + k_0\mu_0\mu_0^T
        # calculate \mu_N\mu_N^T
        mu_n_mu_nT = np.outer(mu_n, mu_n) * k_n

        scaleTdistrn = (k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.))
        scaled_sigma_n = self.prior.sigma + self.sum_squared_table_customers[table_id] - mu_n_mu_nT + self.k0mu0mu0T
        sigma_n = scaled_sigma_n * scaleTdistrn
        # calculate det(Sigma)
        # Use slogdet to avoid under/overflow problems
        sign_det_sig, log_det_sig = slogdet(sigma_n)
        # The sign should always be 1, otherwise we'll run into problems when computing the likelihood
        if sign_det_sig != 1:
            self.log.warn("Error computing determinant of: {}. Table count = {}".format(sigma_n, count))
            raise ValueError("sign of log determinant of sigma is {}".format(sign_det_sig))
        # Now calculate Sigma^(-1) and det(Sigma) and store them
        # calculate Sigma^(-1)
        inv_sigma_n = inv(sigma_n)
        return mu_n, sigma_n, inv_sigma_n, log_det_sig, scaled_sigma_n

    def set_table_parameters(self, table_id):
        """
        Recompute the table parameters for table `table_id` and update their values stored
        as Numpy array attributes.

        This is only used when Cholesky decomposition is disabled

        """
        mu, sigma, inv_sigma, log_det_sig, scaled_sigma = self.calculate_table_params(table_id)
        self.table_means[table_id] = mu
        self.log_determinants[table_id] = log_det_sig
        self.table_inverse_covariances[table_id] = inv_sigma
        return scaled_sigma

    def update_table_params_chol(self, table_id, encoding, is_removed=False):
        count = self.table_counts[table_id]
        k_n = self.prior.kappa + count
        nu_n = self.prior.nu + count
        scaleTdistrn = (k_n + 1.) / (k_n * (float(nu_n) - self.embedding_size + 1.))

        if is_removed:
            # Now use the rank1 downdate to calculate the cholesky decomposition of the updated covariance matrix
            # The update equation is
            #   \Sigma_(N+1) =\Sigma_(N) - (k_0 + N+1) / (k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1}) ^ T
            # Therefore x = sqrt((k_0 + N - 1) / (k_0 + N)) (X_{n} - \mu_{n})
            # Note here \mu_n will be the mean before updating.
            # After updating sigma_n, we will update \mu_n.

            # calculate (X_{n} - \mu_{n-1})
            # This uses the old mean, not yet updated
            x = (encoding - self.table_means[table_id]) * np.sqrt((k_n + 1.) / k_n)
            # The Chol rank1 downdate modifies the array in place
            chol_rank1_downdate(self.table_cholesky_ltriangular_mat[table_id], x)

            # Update the mean
            new_mean = self.table_means[table_id] * (k_n + 1.)
            new_mean -= encoding
            new_mean /= k_n
            self.table_means[table_id] = new_mean
        else:
            # New customer is added
            new_mean = self.table_means[table_id] * (k_n - 1.)
            new_mean += encoding
            new_mean /= k_n
            self.table_means[table_id] = new_mean

            # We need to recompute det(Sig) and (v_{d,i} - mu) . Sig^-1 . (v_{d,i} - mu)
            # v_{d,i} is the word vector being added

            # The rank1 update equation is
            #  \Sigma_{n+1} = \Sigma_{n} + (k_0 + n + 1) / (k_0 + n) * (x_{n+1} - \mu_{n+1})(x_{n+1} - \mu_{n+1}) ^ T
            # calculate (X_{n} - \mu_{n-1})
            # This time we update the mean first and use the new mean
            x = (encoding - self.table_means[table_id]) * np.sqrt(k_n / (k_n - 1.))
            # The update modifies the decomp array in place
            chol_rank1_update(self.table_cholesky_ltriangular_mat[table_id], x)

        # Calculate the 0.5 * log(det) + D / 2 * scaleTdistrn
        # The scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
        self.log_determinants[table_id] = \
            np.sum(np.log(np.diagonal(self.table_cholesky_ltriangular_mat[table_id]))) \
            + self.embedding_size * np.log(scaleTdistrn) / 2.

    def log_multivariate_tdensity(self, x, table_id):
        """
        Density for a single table.

        Permits batching rows of x to compute density for multiple embeddings at once.

        This is for the non-Cholesky mode.

        """
        if self.cholesky_decomp:
            return self._log_multivariate_tdensity_chol(x, table_id)

        mu = self.table_means[table_id]
        sigma_inv = self.table_inverse_covariances[table_id]
        count = self.table_counts[table_id]
        log_detr = self.log_determinants[table_id]

        # Now calculate the likelihood
        # calculate degrees of freedom of the T-distribution
        nu = self.prior.nu + count - self.embedding_size + 1.
        x_minus_mu = x - mu
        # Calculate (x = mu)^TSigma^(-1)(x = mu)
        # vec . mat -> vec
        prod = np.dot(x_minus_mu, sigma_inv)
        # vec . vec -> scalar
        # This is just a dot product, but implemented this way to allow us to batch x
        prod1 = np.sum(prod * x_minus_mu, -1)
        # Should be Nx1
        logprob = gammaln((nu + self.embedding_size) / 2.) - (
                    gammaln(nu / 2.) + self.embedding_size / 2. *
                    (np.log(nu) + np.log(math.pi)) + 0.5 * log_detr + (
                        nu + self.embedding_size) / 2. * np.log(1. + prod1 / nu))
        return logprob

    def log_multivariate_tdensity_tables(self, x):
        """
        Density for all tables in parallel. This version only allows a single x at a time,
        but all tables, as required for sampling.

        This is for the non-Cholesky mode.

        """
        if self.cholesky_decomp:
            return self._log_multivariate_tdensity_chol_tables(x)

        ## Do for table 0 for debugging
        # K x E
        mu = self.table_means
        #mu_0 = self.table_means[0]
        # K x E x E
        sigma_inv = self.table_inverse_covariances
        #sigma_inv_0 = self.table_inverse_covariances[0]
        # K
        count = self.table_counts
        #count_0 = self.table_counts[0]
        # K
        log_detr = self.log_determinants
        #log_detr_0 = self.log_determinants[0]

        # Now calculate the likelihood
        # calculate degrees of freedom of the T-distribution
        nu = self.prior.nu + count - self.embedding_size + 1.  # (K,)'
        #nu_0 = self.prior.nu + count_0 - self.embedding_size + 1.
        x_minus_mu = x[np.newaxis, :] - mu  # (K, E)
        #x_minus_mu_0 = x - mu_0
        # Calculate (x = mu)^TSigma^(-1)(x = mu)
        # vec . mat -> vec (batched)
        prod = np.sum(x_minus_mu[:, :, np.newaxis] * sigma_inv, axis=-1)  # (K, E)
        #prod_0 = np.dot(x_minus_mu_0, sigma_inv_0)
        # vec . vec -> scalar
        # This is just a dot product, but implemented this way to allow us to batch tables
        prod1 = np.sum(prod * x_minus_mu, -1)
        #prod1_0 = np.sum(prod_0 * x_minus_mu_0, -1)
        # (K,) -- one value per table
        logprob = gammaln((nu + self.embedding_size) / 2.) - (
                    gammaln(nu / 2.) + self.embedding_size / 2. *
                    (np.log(nu) + np.log(math.pi)) + 0.5 * log_detr + (
                        nu + self.embedding_size) / 2. * np.log(1. + prod1 / nu))
        return logprob

    def _log_multivariate_tdensity_chol(self, x, table_id):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.

        """
        if x.ndim > 1:
            logprobs = np.zeros(x.shape[0], dtype=np.float64)
            for i in range(x.shape[0]):
                logprobs[i] = self._log_multivariate_tdensity_chol(x[i], table_id)
            return logprobs

        count = self.table_counts[table_id]
        k_n = self.prior.kappa + count
        nu_n = self.prior.nu + count
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))
        nu = self.prior.nu + count - self.embedding_size + 1.
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver
        # first calculate (x-mu)
        x_minus_mu = x - self.table_means[table_id]
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn * self.table_cholesky_ltriangular_mat[table_id]
        solved = solve_triangular(ltriangular_chol, x_minus_mu)
        # Now take xTx (dot product)
        val = (solved ** 2.).sum(-1)

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          self.log_determinants[table_id] +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def _log_multivariate_tdensity_chol_tables(self, x):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.
        This version computes the likelihood for all tables in parallel.

        """
        count = self.table_counts
        k_n = self.prior.kappa + count
        nu_n = self.prior.nu + count
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))
        nu = self.prior.nu + count - self.embedding_size + 1.
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver first calculate (x-mu)
        x_minus_mu = x[None, :] - self.table_means
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn[:, None, None] * self.table_cholesky_ltriangular_mat
        # We can't do solve_triangular for all matrices at once in scipy
        val = np.zeros(self.num_tables, dtype=np.float64)
        for table in range(self.num_tables):
            table_solved = solve_triangular(ltriangular_chol[table], x_minus_mu[table])
            # Now take xTx (dot product)
            val[table] = (table_solved ** 2.).sum()

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          self.log_determinants +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def sample(self, num_iterations):
        """
        for num_iters:
            for each customer
                remove him from his old_table and update the table params.
                if old_table is empty:
                    remove table
                Calculate prior and likelihood for this customer sitting at each table
                sample for a table index
                if new_table is equal to old_table
                    don't have to update the parameters
                else update params of the old table.
        """
        for iteration in range(num_iterations):
            self.log.info("Iteration {}".format(iteration))

            pbar = get_progress_bar(len(self.corpus), title="Sampling")
            for d, doc in enumerate(pbar(self.corpus)):
                # if self.show_topics is not None and self.show_topics > 0 and d % self.show_topics == 0:
                #     print("Topics after {:,} docs".format(d))
                #     print(self.format_topics())

                sentence_encodings = self.encoder.encode(doc)
                for w, x in enumerate(sentence_encodings):

                    # Remove custId from his old_table
                    old_table_id = self.table_assignments[d][w]
                    self.table_assignments[d][w] = -1  # Doesn't really make any difference, as only counts are used
                    self.table_counts[old_table_id] -= 1
                    self.table_counts_per_doc[old_table_id, d] -= 1
                    # Update vector means etc
                    self.sum_table_customers[old_table_id] -= x
                    self.sum_squared_table_customers[old_table_id] -= np.outer(x, x)

                    # Topic 'old_tabe_id' now has one member fewer
                    if self.cholesky_decomp:
                        # Just update params for this customer
                        self.update_table_params_chol(old_table_id, x, is_removed=True)
                    else:
                        # Now recalculate table paramters for this table
                        self.set_table_parameters(old_table_id)

                    #self.check_everything(iteration, d, w, mid_sample=True)

                    # Now calculate the prior and likelihood for the customer to sit in each table and sample
                    # Go over each table
                    counts = self.table_counts_per_doc[:, d] + self.alpha
                    # Now calculate the likelihood for each table
                    log_lls = self.log_multivariate_tdensity_tables(x)
                    # Add log prior in the posterior vector
                    log_posteriors = np.log(counts) + log_lls
                    # To prevent overflow, subtract by log(p_max).
                    # This is because when we will be normalizing after exponentiating,
                    # each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
                    # the log p_max cancels put and prevents overflow in the exponentiating phase.
                    posterior = np.exp(log_posteriors - log_posteriors.max())
                    posterior /= posterior.sum()
                    # Now sample an index from this posterior vector.
                    new_table_id = np.random.choice(self.num_tables, p=posterior)

                    # Now have a new assignment: add its counts
                    self.table_assignments[d][w] = new_table_id
                    self.table_counts[new_table_id] += 1
                    self.table_counts_per_doc[new_table_id, d] += 1
                    self.sum_table_customers[new_table_id] += x
                    self.sum_squared_table_customers[new_table_id] += np.outer(x, x)

                    if self.cholesky_decomp:
                        self.update_table_params_chol(new_table_id, x)
                    else:
                        self.set_table_parameters(new_table_id)

                    #self.check_everything(iteration, d, w)

                #if self.cholesky_decomp:
                #    # After each iteration, recompute the Cholesky decomposition fully, to avoid numerical inaccuracies
                #    # blowing up with the repeated updates
                #    # This also recomputes means
                #    for table in range(self.num_tables):
                #        inv_sigma = self.set_table_parameters(table)
                #        self.table_cholesky_ltriangular_mat[table] = cholesky(inv_sigma)

            self.log.info("Computing average LL")
            ave_ll = calculate_avg_ll(
                get_progress_bar(len(self.corpus))(self.corpus), self.table_assignments, self.encoder,
                self.table_means, self.table_cholesky_ltriangular_mat,
                self.prior, self.table_counts_per_doc
            )
            self.average_ll.append(ave_ll)
            self.log.info("Average LL: {:.3e}".format(ave_ll))

            if self.save_path is not None:
                self.log.info("Saving model")
                self.save()

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)

        with open(os.path.join(self.save_path, "params.json"), "w") as f:
            json.dump({
                "average_ll": self.average_ll,
                "alpha": self.alpha,
                "num_tables": self.num_tables,
                "kappa": self.prior.kappa,
            }, f)
        for name, data in [
            ("table_counts", self.table_counts),
            ("table_counts_per_doc", self.table_counts_per_doc.T),
            ("table_means", self.table_means),
            ("table_inverse_covariances", self.table_inverse_covariances),
            ("log_determinants", self.log_determinants),
            ("sum_table_customers", self.sum_table_customers),
            ("sum_squared_table_customers", self.sum_squared_table_customers),
            ("table_cholesky_ltriangular_mat", self.table_cholesky_ltriangular_mat),
            ("prior_mu", self.prior.mu)
        ]:
            with open(os.path.join(self.save_path, "{}.pkl".format(name)), "wb") as f:
                pickle.dump(data, f)
