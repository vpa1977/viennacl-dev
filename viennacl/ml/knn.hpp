#ifndef VIENNACL_ML_KNN_HPP
#define VIENNACL_ML_KNN_HPP
#include <viennacl/forwards.h>
#include <viennacl/context.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/ml/knn_sliding_window.hpp>

#ifdef VIENNACL_WITH_OPENCL
#include <viennacl/ml/opencl/ml_helpers.hpp>
#endif

namespace viennacl
{
namespace ml
{
namespace knn
{

	class naive_knn
	{
	public:
		naive_knn(viennacl::context& ctx) : context_(ctx) {}


		void calc_distance(viennacl::vector<double>& result,const dense_sliding_window& sliding_window, int start_row, int end_row, const viennacl::vector<double>& sample)
		{
			switch( context_.memory_type())
			{
			case viennacl::MAIN_MEMORY:
				calc_distance_cpu(result,sliding_window,start_row,end_row, sample);
				break;
	#ifdef VIENNACL_WITH_OPENCL
			case viennacl::OPENCL_MEMORY:
				viennacl::ml::opencl::knn_kernels::calc_distance(result,sliding_window,start_row,end_row, sample);
				break;
	#endif
	#ifdef VIENNACL_WITH_HSA
			case viennacl::HSA_MEMORY:
				throw memory_exception("Not implemented");
				break;
	#endif
			default:
				throw memory_exception("Not implemented");

			}
		}

		void update_bounds(dense_sliding_window& sliding_window)
		{
			switch( context_.memory_type())
			{
				case viennacl::MAIN_MEMORY:
					update_bounds_cpu(sliding_window);
					break;
		#ifdef VIENNACL_WITH_OPENCL
				case viennacl::OPENCL_MEMORY:
					throw memory_exception("Not Implemented");
					break;
		#endif
		#ifdef VIENNACL_WITH_HSA
				case viennacl::HSA_MEMORY:
					throw memory_exception("Not implemented");
					break;
		#endif
				default:
					throw memory_exception("Not implemented");

				}
		}
	private:
		void update_bounds_cpu(dense_sliding_window& sliding_window)
		{
			for (size_t column = 0; column < sliding_window.m_values_window.size2(); ++column)
			{
				double cur_max = -1 * (std::numeric_limits<double>::max)();
				double cur_min =  1 * (std::numeric_limits<double>::max)();
				for (size_t row = 0; row < sliding_window.m_values_window.size1() ; ++row)
				{
					double value = sliding_window.m_values_window(row, column);
					if (value > cur_max)
						cur_max = value;
					if (value < cur_min)
						cur_min = value;
				}
				sliding_window.m_max_range(column) = cur_max;
				sliding_window.m_min_range(column) = cur_min;
			}

		}
		void calc_distance_cpu(viennacl::vector<double>& result,const dense_sliding_window& sliding_window, int start_row, int end_row, const viennacl::vector<double>& sample)
		{
			for (int row = start_row ; row <= end_row; ++row)
				result(row) = point_distance(row, sliding_window, sample);
		}

		double point_distance(int row, const dense_sliding_window& sliding_window, const viennacl::vector<double>& sample)
		{
			const viennacl::vector<double>& mins = sliding_window.m_min_range;
			const viennacl::vector<double>& maxs = sliding_window.m_max_range;

			const viennacl::vector<double>& row_vector = viennacl::row(sliding_window.m_values_window, row);
			double distance = 0;
			for (size_t i = 0 ; i < row_vector.size() ; ++i)
			{

				double sample_norm = (row_vector(i) - mins(i))/(maxs(i) - mins(i));
				double example_norm = (sample(i) - mins(i))/(maxs(i) - mins(i));

				double d= (sample_norm - example_norm);
				if (sliding_window.m_attribute_types(i) > 0) // nominal attribute
					d = d != 0 ? 1 : 0;
				distance += d * d;
			}
			return distance;

		}
	private:
		viennacl::context& context_;

	};



}
}
}


#endif
