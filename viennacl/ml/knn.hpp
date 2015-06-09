#ifndef VIENNACL_ML_KNN_HPP
#define VIENNACL_ML_KNN_HPP
#include <viennacl/forwards.h>
#include <viennacl/context.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/maxmin.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/ml/knn_sliding_window.hpp>
#include <memory>

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
					update_bounds_viennacl(sliding_window);
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
		void update_bounds_viennacl(dense_sliding_window& sliding_window)
		{
			for (size_t column = 0; column < sliding_window.m_values_window.size2(); ++column)
			{
				const viennacl::vector<double>& attributes = viennacl::column(sliding_window.m_values_window, column);

				double d = viennacl::linalg::max(attributes);
				sliding_window.m_max_range(column) = d;
				d = viennacl::linalg::min(attributes);
				sliding_window.m_min_range(column) = d;
			}

		}
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
			for (int row = start_row ; row < end_row; ++row)
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


	/**
	 * KD-Tree design follows WEKA KD-tree
	 */
	struct kd_tree_node
	{
		dense_sliding_window data_;
		std::shared_ptr<kd_tree_node> left_;
		std::shared_ptr<kd_tree_node> *right_;
	};

	class kd_tree_node_splitter
	{
	public:
		virtual void split(kd_tree_node& parent) = 0;
	};


	class kd_tree_knn : public naive_knn
	{
	public:
		kd_tree_knn(viennacl::context& ctx) : naive_knn(ctx) {}

		void build_tree(const dense_sliding_window& sliding_window)
		{

		}



	};



}
}
}


#endif
