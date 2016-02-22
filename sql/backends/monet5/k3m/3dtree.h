/**************************************************************************
 * This file is part of K3Match.                                          *
 * Copyright (C) 2016 Pim Schellart <P.Schellart@astro.ru.nl>             *
 *                                                                        *
 * This Source Code Form is subject to the terms of the Mozilla Public    *
 * License, v. 2.0. If a copy of the MPL was not distributed with this    *
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.               *
 **************************************************************************/

#ifndef __K3MATCH_3DTREE_H__
#define __K3MATCH_3DTREE_H__

#include <k3match.h>

typedef struct node_t node_t;

struct node_t {
  int axis;

  point_t* point;

  node_t *parent, *left, *right;
};

void k3m_build_balanced_tree(node_t *tree, point_t **points, int_t npoints, int axis, int_t *npool);

void k3m_print_tree(node_t *tree);

void k3m_print_dot_tree(node_t *tree);

node_t* k3m_insert_node(node_t *tree, node_t *node);

node_t* k3m_closest_leaf(node_t *tree, point_t *point);

node_t* k3m_nearest_neighbour(node_t *tree, point_t *point);

int_t k3m_in_range(node_t *tree, point_t **match, point_t *search, real_t ds);

#endif // __K3MATCH_3DTREE_H__

