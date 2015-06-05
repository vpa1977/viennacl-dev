/*
 * knn_sliding_window.hpp
 *
 *  Created on: 2/06/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_KNN_SLIDING_WINDOW_HPP_
#define VIENNACL_ML_KNN_SLIDING_WINDOW_HPP_

#include <viennacl/context.hpp>

namespace viennacl
{
namespace ml
{
namespace knn
{
	struct dense_sliding_window
	{
		dense_sliding_window(viennacl::context& ctx, int num_attributes, int window_size) :
			    m_row_index(0),
				m_attribute_types(num_attributes,ctx),
				m_values_window(window_size,num_attributes,ctx),
				m_classes_values(window_size,ctx),
				m_min_range(num_attributes,ctx),
				m_max_range(num_attributes,ctx),
				m_context(ctx)  {}

		int m_row_index;
		viennacl::vector<int> m_attribute_types;
		viennacl::matrix<double> m_values_window;
		viennacl::vector<double> m_classes_values;
		viennacl::vector<double> m_min_range;
		viennacl::vector<double> m_max_range;

	private:
		viennacl::context& m_context;
	};
}
}
}


#endif /* VIENNACL_ML_KNN_SLIDING_WINDOW_HPP_ */
