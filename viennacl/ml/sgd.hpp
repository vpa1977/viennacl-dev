/*
 * sgd.hpp
 *
 *  Created on: 13/04/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_SGD_HPP_
#define VIENNACL_ML_SGD_HPP_

#include <viennacl/forwards.h>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/scalar.hpp>

#include <viennacl/linalg/vector_operations.hpp>
#include <viennacl/linalg/matrix_operations.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/prod.hpp>       //generic matrix-vector product
#include <viennacl/linalg/host_based/common.hpp>
#include <viennacl/context.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/ml/weights_update.hpp>

namespace viennacl {
namespace ml {
/**
 * SGD implementation using
 */
template< typename sgd_matrix_type>
class sgd_template {
public:
	enum LossFunction {
		HINGE, LOGLOSS, SQUAREDLOSS
	};
	sgd_template(size_t len, LossFunction loss,viennacl::context& ctx, bool nominal = true ) :
			loss_(loss), context_(ctx), weights_(len, ctx) {
		learning_rate_ = 0.01;
		lambda_ = 0.0001;
		nominal_ = nominal;
		instance_size_ = len;
	}

	size_t instance_size()
	{
		return instance_size_;
	}

	void print_warning()
	{
		printf("Warning: not implemented\n");
	}

	void reset() {
		weights_.clear();
		bias_ = 0;
	}

	std::vector<double> get_votes_for_instance(
			const viennacl::vector<double>& single_instance) {
		switch (context_.memory_type()) {
		case viennacl::MAIN_MEMORY:
			return get_votes_for_instance_cpu(single_instance);
			break;
#ifdef VIENNACL_WITH_OPENCL
		case viennacl::OPENCL_MEMORY:
			break;
#endif
#ifdef VIENNACL_WITH_HSA
		case viennacl::HSA_MEMORY:
			break;
#endif
		default:
			assert(0);

		}
		return std::vector<double>();
	}

	void train(const viennacl::vector<double>& class_values,
			const sgd_matrix_type& batch) {
		assert(batch.size2() == weights_.size());
		viennacl::vector<double> result = viennacl::linalg::prod(batch,	weights_);

		switch (context_.memory_type()) {
		case viennacl::MAIN_MEMORY:
			decay_weights_cpu(class_values.size());
			update_weights_cpu(nominal_, class_values, result, batch);

			break;
#ifdef VIENNACL_WITH_OPENCL
		case viennacl::OPENCL_MEMORY:
			decay_weights_cpu(class_values.size());
			update_weights_opencl(nominal_, class_values, result, batch);
			break;
#endif
#ifdef VIENNACL_WITH_HSA
		case viennacl::HSA_MEMORY:
			decay_weights_cpu(class_values.size());
			update_weights_cpu(nominal_, class_values, result, batch);
			break;
#endif
		default:
			throw memory_exception("not implemented");

		}
	}

	void decay_weights_cpu(size_t batch_size) {
		double multiplier = 1.0 - (learning_rate_ * lambda_) / batch_size;
		weights_ = weights_ * multiplier;
	}

	void update_weights_cpu(bool nominal,
			const viennacl::vector<double>& class_values,
			const viennacl::vector<double>& prod_result,
			const sgd_matrix_type& batch) {
		viennacl::vector<double> factors(class_values.size(), viennacl::context(viennacl::MAIN_MEMORY));
		const double* class_values_ptr =
				viennacl::linalg::host_based::detail::extract_raw_pointer<double>(
						class_values);
		double * factors_ptr =
				viennacl::linalg::host_based::detail::extract_raw_pointer<double>(
						factors);

		size_t size_class = viennacl::traits::size(class_values);
		size_t start_class = viennacl::traits::start(class_values);
		size_t stride_class = viennacl::traits::stride(class_values);

		size_t size_factor = viennacl::traits::size(prod_result);
		size_t start_factor = viennacl::traits::start(prod_result);
		size_t stride_factor = viennacl::traits::stride(prod_result);

		assert(size_factor == size_class);

		size_t size = size_class;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
		for (long i = 0; i < static_cast<long>(size); ++i) {
			size_t class_offset = i * stride_class + start_class;
			size_t prod_offset = i * stride_factor + start_factor;
			double y;
			double z;
			if (nominal_) {
				y = class_values_ptr[class_offset] ? 1 : -1;
				z = y * (prod_result[prod_offset] + bias_);
			} else {
				y = class_values_ptr[class_offset];
				z = y - (prod_result[prod_offset] + bias_);

			}
			factors_ptr[i] = 0;
			if (loss_ != HINGE || (z < 1)) {
				double loss = 0;
				switch (loss_) {
				case HINGE:
					loss = (z < 1) ? 1 : 0;
					break;
				case LOGLOSS:
					if (z < 0) {
						loss = 1.0 / (exp(z) + 1.0);
					} else {
						double t = exp(-z);
						loss = t / (t + 1);
					}
					break;
				case SQUAREDLOSS:
				default:
					assert(0);
				}
				factors_ptr[i] = learning_rate_ * y * loss;
			}
		}

		sgd_update_weights<double>(weights_, batch, factors);

#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
		for (long i = 0; i < static_cast<long>(size); ++i) {
			if (factors_ptr[i]) {
				const viennacl::vector<double> row = viennacl::row(batch, i);
				viennacl::vector<double> row_mult = factors_ptr[i] * row;
				weights_ += row_mult;
			}
		}

		double sum = 0;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
		for (long i = 0; i < static_cast<long>(size); ++i) {
			sum += factors_ptr[i];
		}
		bias_ += sum;
	}

#ifdef VIENNACL_WITH_OPENCL
	void update_weights_opencl(bool nominal,
			const viennacl::vector<double>& class_values,
			const viennacl::vector<double>& prod_result,
			const sgd_matrix_type& batch) {

		viennacl::vector<double> factors(class_values.size(), get_global_context());
		sgd_compute_factors<double>(class_values, prod_result,factors, nominal,(int) loss_, learning_rate_, bias_);
		sgd_update_weights<double>(weights_, batch, factors);

		double sum = 0;
/*		for (long i = 0; i < static_cast<long>(size); ++i) {
			sum += factors[i];
		}*/
		bias_ += sum;
	}

#endif
#ifdef VIENNACL_WITH_HSA

#endif

	std::vector<double> get_votes_for_instance_cpu(
			const viennacl::vector<double>& instance) {

		std::vector<double> res;

		double wx =  viennacl::linalg::inner_prod(weights_, instance);
		double z = wx + bias_;
		if (!nominal_)
			res.push_back(z);
		else {
			if (z <= 0) {
				if (loss_ == LOGLOSS) {
					double prediction = 1.0 / (1.0 + exp(z));
					res.push_back(prediction);
					res.push_back(1 - prediction);
				} else {
					res.push_back(1);
					res.push_back(0);
				}
			} else {
				if (loss_ == LOGLOSS) {
					double prediction = 1.0 / (1.0 + exp(-z));
					res.push_back(prediction);
					res.push_back(1 - prediction);
				} else {
					res.push_back(0);
					res.push_back(1);
				}
			}
		}
		return res;

	}
	const bool is_nominal() const { return nominal_; }
private:
	LossFunction loss_;
	viennacl::context& context_;
	viennacl::vector<double> weights_;
	double learning_rate_;
	double lambda_;
	double bias_;
	bool nominal_;
	size_t instance_size_;
};

typedef sgd_template<viennacl::compressed_matrix<double> > sgd;


}
;
}
;

#endif /* SGD_HPP_ */

