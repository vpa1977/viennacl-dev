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

#include <atomic>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

namespace viennacl {
namespace ml {

class dense_sgd
{
public:
	enum LossFunction {
		HINGE, LOGLOSS, SQUAREDLOSS
	};
	dense_sgd(size_t len, size_t batch_size, LossFunction loss, viennacl::context& ctx, double learning_rate, double lambda, bool nominal = true ) :
			loss_(loss), context_(ctx), weights_(len, ctx), factor_(0), prod_result_(0) {
		learning_rate_ = learning_rate;
		lambda_ = lambda;
		nominal_ = nominal;
		instance_size_ = len;
		reset();
	}

	size_t instance_size()
	{
		return instance_size_;
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
			return get_votes_for_instance_cpu(single_instance);
			break;
#endif
#ifdef VIENNACL_WITH_HSA
		case viennacl::HSA_MEMORY:
			return get_votes_for_instance_cpu(single_instance);
			break;
#endif
		default:
			throw memory_exception("Not implemented");

		}
		return std::vector<double>();
	}

	void train(const std::vector<double>& class_values, const viennacl::vector<double> & batch) {

		using namespace std::chrono;

		for (size_t row = 0; row < class_values.size(); ++row)
		{
			int offset = row*weights_.size();
			viennacl::range r(offset, offset+weights_.size());

			viennacl::vector_range< viennacl::vector<double> > next(batch, r);
			//system_clock::time_point t1 = system_clock::now();
			prod_result_ = viennacl::linalg::inner_prod( next ,	weights_);
			//system_clock::time_point t2 = system_clock::now();
			switch (context_.memory_type()) {
			case viennacl::HSA_MEMORY:
			case viennacl::OPENCL_MEMORY:
			case viennacl::MAIN_MEMORY:
			{
				//system_clock::time_point t3 = system_clock::now();
				decay_weights_cpu(class_values.size());
			//	system_clock::time_point t4 = system_clock::now();
				//viennacl::ml::opencl::dense_sgd_update_weights<double>(row, nominal_, learning_rate_, bias_, loss_, class_values, prod_result_, next, weights_);
				update_weights_cpu(nominal_, class_values, prod_result_, next, row);
		//		system_clock::time_point t5 = system_clock::now();

				/*duration<double> time_span1 = duration_cast<duration<double>>(t2 - t1);
				duration<double> time_span2 = duration_cast<duration<double>>(t4 - t3);
				duration<double> time_span3 = duration_cast<duration<double>>(t5 - t4);

				std::cout << "Times : " << time_span1.count() << " " << time_span2.count() << " " << time_span3.count() << std::endl;
				*/
			}
				break;
			default:
				throw memory_exception("not implemented");

			}


		}

	}

	void decay_weights_cpu(size_t batch_size) {
		double multiplier = 1.0 - (learning_rate_ * lambda_) / batch_size;
		weights_ = weights_ * multiplier;
	}



	void update_weights_cpu(bool nominal,const std::vector<double>& class_values,const viennacl::scalar<double>& prod_result, const viennacl::vector<double>& row_values, int row) {
		double y;
		viennacl::vector<double> z(0,viennacl::traits::context(prod_result));
		if (nominal) {
			y = class_values.at(row) ? 1 : -1;
			z = y * (prod_result + bias_);
		} else {
			y = class_values.at(row);
			z = y - (prod_result + bias_);
			y = 1;

		}
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
			factor_ = learning_rate_ * y * loss;
			bias_ += factor_;
		}
		else
		{
			factor_ =0;
		}
		weights_+= factor_ * row_values;
	}


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

	void print_weights()
	{
		std::cout << "weights: ";
		for (int i = 0; i < weights_.size(); ++i)
			std::cout << weights_(i) << " ";
		std::cout << std::endl;
	}
private:
	LossFunction loss_;
	viennacl::context& context_;
	viennacl::vector<double> weights_;
	double factor_;
	double prod_result_;
	double learning_rate_;
	double lambda_;
	double bias_;
	bool nominal_;
	size_t instance_size_;
};




/**
 * SGD implementation using
 */
template< typename sgd_matrix_type>
class sgd_template {
public:
	enum LossFunction {
		HINGE, LOGLOSS, SQUAREDLOSS
	};
	sgd_template(size_t len, size_t batch_size, LossFunction loss, viennacl::context& ctx, double learning_rate, double lambda, bool nominal = true ) :
		loss_(loss), context_(ctx), weights_(len, ctx), factors_(batch_size, ctx), prod_result_(batch_size, ctx), sum_(0, ctx), bias_(0, ctx), all_ones_(viennacl::scalar_vector<double>(batch_size, 1, ctx)) {
		learning_rate_ = learning_rate;
		lambda_ = lambda;
		nominal_ = nominal;
		instance_size_ = len;
		reset();
	}

	size_t instance_size()
	{
		return instance_size_;
	}


	void reset() {
		weights_.clear();
		bias_ = 0;
	}

	void print_weights()
	{
		std::cout << "weights: ";
		for (int i = 0; i < weights_.size(); ++i)
			std::cout << weights_(i) << " ";
		std::cout << std::endl;
	}

	std::vector<double> get_votes_for_instance(
			const viennacl::vector<double>& single_instance) {
		switch (context_.memory_type()) {
		case viennacl::MAIN_MEMORY:
			return get_votes_for_instance_cpu(single_instance);
			break;
#ifdef VIENNACL_WITH_OPENCL
		case viennacl::OPENCL_MEMORY:
			return get_votes_for_instance_cpu(single_instance);
			break;
#endif
#ifdef VIENNACL_WITH_HSA
		case viennacl::HSA_MEMORY:
			return get_votes_for_instance_cpu(single_instance);
			break;
#endif
		default:
			throw memory_exception("Not implemented");

		}
		return std::vector<double>();
	}

	void train(const viennacl::vector<double>& class_values,
			const sgd_matrix_type& batch) {
		assert(batch.size2() == weights_.size());
		prod_result_ = viennacl::linalg::prod(batch,	weights_);

	//	std::cout << "wx = " << prod_result_ << std::endl;

		switch (context_.memory_type()) {
		case viennacl::MAIN_MEMORY:
			decay_weights_cpu(class_values.size());
			update_weights_cpu(nominal_, class_values, prod_result_, batch);

			break;
#ifdef VIENNACL_WITH_OPENCL
		case viennacl::OPENCL_MEMORY:
			decay_weights_cpu(class_values.size());
			update_weights_opencl(nominal_, class_values, prod_result_, batch);
			break;
#endif
#ifdef VIENNACL_WITH_HSA
		case viennacl::HSA_MEMORY:
			decay_weights_cpu(class_values.size());
			update_weights_cpu(nominal_, class_values, prod_result_, batch);
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


	void compute_factors_cpu(const viennacl::vector<double>&  class_values,
							 const viennacl::vector<double>&  prod_result,
							 viennacl::vector<double>&  factors,
							 bool nominal,
							 LossFunction loss_function,
							 double learning_rate,
							 double bias)
	{
		const double* class_values_ptr =
				viennacl::linalg::host_based::detail::extract_raw_pointer<double>(
						class_values);
		if (factors.size() == 0)
			factors = viennacl::vector<double>(class_values.size(), viennacl::traits::context(class_values));
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
			if (nominal) {
				y = class_values_ptr[class_offset] ? 1 : -1;
				z = y * (prod_result[prod_offset] + bias);
			} else {
				y = class_values_ptr[class_offset];
				z = y - (prod_result[prod_offset] + bias);

			}
			factors_ptr[i] = 0;
			if (loss_function != HINGE || (z < 1)) {
				double loss = 0;
				switch (loss_function) {
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
				factors_ptr[i] = learning_rate * y * loss;
			}
		}

	}

	void update_weights_cpu(bool nominal,
			const viennacl::vector<double>& class_values,
			const viennacl::vector<double>& prod_result,
			const sgd_matrix_type& batch) {

		compute_factors_cpu(class_values, prod_result,factors_, nominal,loss_, learning_rate_, bias_);
		//std::cout << "factor " << factors_ << " learning rate " << learning_rate_ << " bias "<< bias_ << std::endl;
		sgd_update_weights<sgd_matrix_type>(weights_, batch, factors_);
		viennacl::linalg::inner_prod_impl(factors_, all_ones_, sum_);
		bias_ += sum_;
	}

#ifdef VIENNACL_WITH_OPENCL
	void update_weights_opencl(bool nominal,
			const viennacl::vector<double>& class_values,
			const viennacl::vector<double>& prod_result,
			const sgd_matrix_type& batch) {

/*		std::vector<double> _tmp_classes(class_values.size());
		std::vector<double> _tmp_prod_results(prod_result.size());
		std::vector<double> _tmp_cpu_factors(factors_.size());

		viennacl::context mem(viennacl::MAIN_MEMORY);
		viennacl::vector<double> cpu_classes(class_values.size(), mem);
		viennacl::vector<double> cpu_prod_result(prod_result.size(), mem);
		viennacl::vector<double> cpu_factors(factors_.size(), mem);

		viennacl::copy(class_values, _tmp_classes);
		viennacl::copy(prod_result, _tmp_prod_results);
		viennacl::copy(factors_, _tmp_cpu_factors);

		viennacl::copy(_tmp_classes, cpu_classes);
		viennacl::copy(_tmp_prod_results, cpu_prod_result);
		viennacl::copy(_tmp_cpu_factors, cpu_factors);

		compute_factors_cpu( cpu_classes, cpu_prod_result, cpu_factors, nominal, loss_, learning_rate_, bias_);*/

		sgd_compute_factors<double>(class_values, prod_result,factors_, nominal,(int) loss_, learning_rate_, bias_);

		/*for (int i = 0 ;i < factors_.size() ; ++i)
		{
			if (factors_(i)!= cpu_factors(i))
			{
				std::cout << "expected " << cpu_factors(i) << " got "<< factors_(i) << std::endl;
				std::cout << "b" << std::endl;
			}
		}*/

		sgd_update_weights<sgd_matrix_type>(weights_, batch, factors_);
		viennacl::linalg::inner_prod_impl(factors_, all_ones_, sum_);
		bias_ += sum_;
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
	viennacl::vector<double> factors_;
	viennacl::vector<double> prod_result_;
	double learning_rate_;
	double lambda_;
	viennacl::scalar<double> bias_;
	viennacl::scalar<double> sum_;
	viennacl::vector<double> all_ones_; ;
	bool nominal_;
	size_t instance_size_;

};



typedef sgd_template<viennacl::compressed_matrix<double> > sgd;


}
;
}
;

#endif /* SGD_HPP_ */

