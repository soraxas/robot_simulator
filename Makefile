.PHONY: all motion_bench clean

ROBOT_RESOURCES=robot_resources
ROBOT_RESOURCES_SRC=robot_resources/_src

GH_RELEASE_URL_ROOT=https://github.com/soraxas/robot_simulator/releases/download


NEEDED=$(ROBOT_RESOURCES)/robot_model/kinova/urdf/jaco_clean.urdf $(ROBOT_RESOURCES)/dataset_traj-demonstrations/clfd/robot_hello_world/helloworld_names.json

all: $(NEEDED)

# robot dataset clfd
$(ROBOT_RESOURCES)/dataset_traj-demonstrations/clfd/robot_hello_world/helloworld_names.json: $(ROBOT_RESOURCES_SRC)/traj-demo-clfd/dataset_traj-demonstrations.tar.xz
	mkdir -p $(ROBOT_RESOURCES)/dataset_traj-demonstrations
	tar xvf $^ -C $(ROBOT_RESOURCES)
	@touch $@

# robot model
$(ROBOT_RESOURCES)/robot_model/kinova/urdf/jaco_clean.urdf: $(ROBOT_RESOURCES_SRC)/robot_model/robot_model.tar.xz
	mkdir -p $(ROBOT_RESOURCES)/robot_model
	tar xvf $^ -C $(ROBOT_RESOURCES)
	@touch $@


# download missing file from github release
$(ROBOT_RESOURCES_SRC)/%.tar.xz:
	mkdir -p $(dir $@)
	cd $(dir $@) && wget "$(GH_RELEASE_URL_ROOT)/$*.tar.xz"

#############################################################
# for robot learning resources
include Makefile.needed

HOST=https://raw.githubusercontent.com/soraxas/__file-store__/master
DATA_DIR=robodata/motion_bench

motion_bench:  $(addprefix $(DATA_DIR)/, $(NEEDED))

$(DATA_DIR)/%:
	mkdir -p "$(DATA_DIR)"
	# wget  --directory-prefix="$(DATA_DIR)" "$(HOST)/stein/kitchen/$*"
	wget  --directory-prefix="$(DATA_DIR)" "$(HOST)/stein/$*"


