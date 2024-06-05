idf_component_register(SRC_DIRS          "."
                       PRIV_INCLUDE_DIRS  "." "${ESP_MATTER_PATH}/examples/common/utils" "${CMAKE_CURRENT_LIST_DIR}/include"
                       EMBED_FILES diagnostic_logs/end_user_support.log diagnostic_logs/network_diag.log)

set_property(TARGET ${COMPONENT_LIB} PROPERTY CXX_STANDARD 17)
target_compile_options(${COMPONENT_LIB} PRIVATE "-DCHIP_HAVE_CONFIG_H")
