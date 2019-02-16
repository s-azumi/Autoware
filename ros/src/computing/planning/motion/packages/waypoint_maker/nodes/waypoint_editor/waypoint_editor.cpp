/*
 * Copyright 2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * v1.0 Masaya Kataoka
 */

#include "waypoint_editor.h"

WaypointEditor::WaypointEditor(ros::NodeHandle nh,ros::NodeHandle pnh)
{
    server_ptr_ = boost::make_shared<interactive_markers::InteractiveMarkerServer>("waypoint_editor");
    WaypointHandlerMarker marker(server_ptr_,0);
    markers_.push_back(marker);
}

WaypointEditor::~WaypointEditor()
{

}