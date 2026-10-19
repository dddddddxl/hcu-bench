#include <hip/hip_runtime.h>


// Kernel 1: write v16..v255 via inline asm and busy-loop with SGPR ops
__launch_bounds__(256, 1)
__global__ void kernel1_write_vgprs_and_busy_safe(int busy_loops1, uint32_t *output, int VGPR_COUNT) {
    asm volatile("" ::: "memory");
    uint32_t temp;
    asm volatile(
        "v_mov_b32 v16, 2\n\t"
        "v_mov_b32 v17, 2\n\t"
        "v_mov_b32 v18, 2\n\t"
        "v_mov_b32 v19, 2\n\t"
        "v_mov_b32 v20, 2\n\t"
        "v_mov_b32 v21, 2\n\t"
        "v_mov_b32 v22, 2\n\t"
        "v_mov_b32 v23, 2\n\t"
        "v_mov_b32 v24, 2\n\t"
        "v_mov_b32 v25, 2\n\t"
        "v_mov_b32 v26, 2\n\t"
        "v_mov_b32 v27, 2\n\t"
        "v_mov_b32 v28, 2\n\t"
        "v_mov_b32 v29, 2\n\t"
        "v_mov_b32 v30, 2\n\t"
        "v_mov_b32 v31, 2\n\t"
        "v_mov_b32 v32, 2\n\t"
        "v_mov_b32 v33, 2\n\t"
        "v_mov_b32 v34, 2\n\t"
        "v_mov_b32 v35, 2\n\t"
        "v_mov_b32 v36, 2\n\t"
        "v_mov_b32 v37, 2\n\t"
        "v_mov_b32 v38, 2\n\t"
        "v_mov_b32 v39, 2\n\t"
        "v_mov_b32 v40, 2\n\t"
        "v_mov_b32 v41, 2\n\t"
        "v_mov_b32 v42, 2\n\t"
        "v_mov_b32 v43, 2\n\t"
        "v_mov_b32 v44, 2\n\t"
        "v_mov_b32 v45, 2\n\t"
        "v_mov_b32 v46, 2\n\t"
        "v_mov_b32 v47, 2\n\t"
        "v_mov_b32 v48, 2\n\t"
        "v_mov_b32 v49, 2\n\t"
        "v_mov_b32 v50, 2\n\t"
        "v_mov_b32 v51, 2\n\t"
        "v_mov_b32 v52, 2\n\t"
        "v_mov_b32 v53, 2\n\t"
        "v_mov_b32 v54, 2\n\t"
        "v_mov_b32 v55, 2\n\t"
        "v_mov_b32 v56, 2\n\t"
        "v_mov_b32 v57, 2\n\t"
        "v_mov_b32 v58, 2\n\t"
        "v_mov_b32 v59, 2\n\t"
        "v_mov_b32 v60, 2\n\t"
        "v_mov_b32 v61, 2\n\t"
        "v_mov_b32 v62, 2\n\t"
        "v_mov_b32 v63, 2\n\t"
        "v_mov_b32 v64, 2\n\t"
        "v_mov_b32 v65, 2\n\t"
        "v_mov_b32 v66, 2\n\t"
        "v_mov_b32 v67, 2\n\t"
        "v_mov_b32 v68, 2\n\t"
        "v_mov_b32 v69, 2\n\t"
        "v_mov_b32 v70, 2\n\t"
        "v_mov_b32 v71, 2\n\t"
        "v_mov_b32 v72, 2\n\t"
        "v_mov_b32 v73, 2\n\t"
        "v_mov_b32 v74, 2\n\t"
        "v_mov_b32 v75, 2\n\t"
        "v_mov_b32 v76, 2\n\t"
        "v_mov_b32 v77, 2\n\t"
        "v_mov_b32 v78, 2\n\t"
        "v_mov_b32 v79, 2\n\t"
        "v_mov_b32 v80, 2\n\t"
        "v_mov_b32 v81, 2\n\t"
        "v_mov_b32 v82, 2\n\t"
        "v_mov_b32 v83, 2\n\t"
        "v_mov_b32 v84, 2\n\t"
        "v_mov_b32 v85, 2\n\t"
        "v_mov_b32 v86, 2\n\t"
        "v_mov_b32 v87, 2\n\t"
        "v_mov_b32 v88, 2\n\t"
        "v_mov_b32 v89, 2\n\t"
        "v_mov_b32 v90, 2\n\t"
        "v_mov_b32 v91, 2\n\t"
        "v_mov_b32 v92, 2\n\t"
        "v_mov_b32 v93, 2\n\t"
        "v_mov_b32 v94, 2\n\t"
        "v_mov_b32 v95, 2\n\t"
        "v_mov_b32 v96, 2\n\t"
        "v_mov_b32 v97, 2\n\t"
        "v_mov_b32 v98, 2\n\t"
        "v_mov_b32 v99, 2\n\t"
        "v_mov_b32 v100, 2\n\t"
        "v_mov_b32 v101, 2\n\t"
        "v_mov_b32 v102, 2\n\t"
        "v_mov_b32 v103, 2\n\t"
        "v_mov_b32 v104, 2\n\t"
        "v_mov_b32 v105, 2\n\t"
        "v_mov_b32 v106, 2\n\t"
        "v_mov_b32 v107, 2\n\t"
        "v_mov_b32 v108, 2\n\t"
        "v_mov_b32 v109, 2\n\t"
        "v_mov_b32 v110, 2\n\t"
        "v_mov_b32 v111, 2\n\t"
        "v_mov_b32 v112, 2\n\t"
        "v_mov_b32 v113, 2\n\t"
        "v_mov_b32 v114, 2\n\t"
        "v_mov_b32 v115, 2\n\t"
        "v_mov_b32 v116, 2\n\t"
        "v_mov_b32 v117, 2\n\t"
        "v_mov_b32 v118, 2\n\t"
        "v_mov_b32 v119, 2\n\t"
        "v_mov_b32 v120, 2\n\t"
        "v_mov_b32 v121, 2\n\t"
        "v_mov_b32 v122, 2\n\t"
        "v_mov_b32 v123, 2\n\t"
        "v_mov_b32 v124, 2\n\t"
        "v_mov_b32 v125, 2\n\t"
        "v_mov_b32 v126, 2\n\t"
        "v_mov_b32 v127, 2\n\t"
        "v_mov_b32 v128, 2\n\t"
        "v_mov_b32 v129, 2\n\t"
        "v_mov_b32 v130, 2\n\t"
        "v_mov_b32 v131, 2\n\t"
        "v_mov_b32 v132, 2\n\t"
        "v_mov_b32 v133, 2\n\t"
        "v_mov_b32 v134, 2\n\t"
        "v_mov_b32 v135, 2\n\t"
        "v_mov_b32 v136, 2\n\t"
        "v_mov_b32 v137, 2\n\t"
        "v_mov_b32 v138, 2\n\t"
        "v_mov_b32 v139, 2\n\t"
        "v_mov_b32 v140, 2\n\t"
        "v_mov_b32 v141, 2\n\t"
        "v_mov_b32 v142, 2\n\t"
        "v_mov_b32 v143, 2\n\t"
        "v_mov_b32 v144, 2\n\t"
        "v_mov_b32 v145, 2\n\t"
        "v_mov_b32 v146, 2\n\t"
        "v_mov_b32 v147, 2\n\t"
        "v_mov_b32 v148, 2\n\t"
        "v_mov_b32 v149, 2\n\t"
        "v_mov_b32 v150, 2\n\t"
        "v_mov_b32 v151, 2\n\t"
        "v_mov_b32 v152, 2\n\t"
        "v_mov_b32 v153, 2\n\t"
        "v_mov_b32 v154, 2\n\t"
        "v_mov_b32 v155, 2\n\t"
        "v_mov_b32 v156, 2\n\t"
        "v_mov_b32 v157, 2\n\t"
        "v_mov_b32 v158, 2\n\t"
        "v_mov_b32 v159, 2\n\t"
        "v_mov_b32 v160, 2\n\t"
        "v_mov_b32 v161, 2\n\t"
        "v_mov_b32 v162, 2\n\t"
        "v_mov_b32 v163, 2\n\t"
        "v_mov_b32 v164, 2\n\t"
        "v_mov_b32 v165, 2\n\t"
        "v_mov_b32 v166, 2\n\t"
        "v_mov_b32 v167, 2\n\t"
        "v_mov_b32 v168, 2\n\t"
        "v_mov_b32 v169, 2\n\t"
        "v_mov_b32 v170, 2\n\t"
        "v_mov_b32 v171, 2\n\t"
        "v_mov_b32 v172, 2\n\t"
        "v_mov_b32 v173, 2\n\t"
        "v_mov_b32 v174, 2\n\t"
        "v_mov_b32 v175, 2\n\t"
        "v_mov_b32 v176, 2\n\t"
        "v_mov_b32 v177, 2\n\t"
        "v_mov_b32 v178, 2\n\t"
        "v_mov_b32 v179, 2\n\t"
        "v_mov_b32 v180, 2\n\t"
        "v_mov_b32 v181, 2\n\t"
        "v_mov_b32 v182, 2\n\t"
        "v_mov_b32 v183, 2\n\t"
        "v_mov_b32 v184, 2\n\t"
        "v_mov_b32 v185, 2\n\t"
        "v_mov_b32 v186, 2\n\t"
        "v_mov_b32 v187, 2\n\t"
        "v_mov_b32 v188, 2\n\t"
        "v_mov_b32 v189, 2\n\t"
        "v_mov_b32 v190, 2\n\t"
        "v_mov_b32 v191, 2\n\t"
        "v_mov_b32 v192, 2\n\t"
        "v_mov_b32 v193, 2\n\t"
        "v_mov_b32 v194, 2\n\t"
        "v_mov_b32 v195, 2\n\t"
        "v_mov_b32 v196, 2\n\t"
        "v_mov_b32 v197, 2\n\t"
        "v_mov_b32 v198, 2\n\t"
        "v_mov_b32 v199, 2\n\t"
        "v_mov_b32 v200, 2\n\t"
        "v_mov_b32 v201, 2\n\t"
        "v_mov_b32 v202, 2\n\t"
        "v_mov_b32 v203, 2\n\t"
        "v_mov_b32 v204, 2\n\t"
        "v_mov_b32 v205, 2\n\t"
        "v_mov_b32 v206, 2\n\t"
        "v_mov_b32 v207, 2\n\t"
        "v_mov_b32 v208, 2\n\t"
        "v_mov_b32 v209, 2\n\t"
        "v_mov_b32 v210, 2\n\t"
        "v_mov_b32 v211, 2\n\t"
        "v_mov_b32 v212, 2\n\t"
        "v_mov_b32 v213, 2\n\t"
        "v_mov_b32 v214, 2\n\t"
        "v_mov_b32 v215, 2\n\t"
        "v_mov_b32 v216, 2\n\t"
        "v_mov_b32 v217, 2\n\t"
        "v_mov_b32 v218, 2\n\t"
        "v_mov_b32 v219, 2\n\t"
        "v_mov_b32 v220, 2\n\t"
        "v_mov_b32 v221, 2\n\t"
        "v_mov_b32 v222, 2\n\t"
        "v_mov_b32 v223, 2\n\t"
        "v_mov_b32 v224, 2\n\t"
        "v_mov_b32 v225, 2\n\t"
        "v_mov_b32 v226, 2\n\t"
        "v_mov_b32 v227, 2\n\t"
        "v_mov_b32 v228, 2\n\t"
        "v_mov_b32 v229, 2\n\t"
        "v_mov_b32 v230, 2\n\t"
        "v_mov_b32 v231, 2\n\t"
        "v_mov_b32 v232, 2\n\t"
        "v_mov_b32 v233, 2\n\t"
        "v_mov_b32 v234, 2\n\t"
        "v_mov_b32 v235, 2\n\t"
        "v_mov_b32 v236, 2\n\t"
        "v_mov_b32 v237, 2\n\t"
        "v_mov_b32 v238, 2\n\t"
        "v_mov_b32 v239, 2\n\t"
        "v_mov_b32 v240, 2\n\t"
        "v_mov_b32 v241, 2\n\t"
        "v_mov_b32 v242, 2\n\t"
        "v_mov_b32 v243, 2\n\t"
        "v_mov_b32 v244, 2\n\t"
        "v_mov_b32 v245, 2\n\t"
        "v_mov_b32 v246, 2\n\t"
        "v_mov_b32 v247, 2\n\t"
        "v_mov_b32 v248, 2\n\t"
        "v_mov_b32 v249, 2\n\t"
        "v_mov_b32 v250, 2\n\t"
        "v_mov_b32 v251, 2\n\t"
        "v_mov_b32 v252, 2\n\t"
        "v_mov_b32 v253, 2\n\t"
        "v_mov_b32 v254, 2\n\t"
        "v_mov_b32 %0, 2\n\t"
        : "={v255}"(temp)
        :
        : "memory"
    );

    for (int i = 0; i < busy_loops1; ++i) {
        asm volatile(
            "s_add_u32 s0, s0, 1\n\t"
            "s_nop 0\n\t"
            ::: "s0"
        );
    }
    
    asm volatile("" ::: "memory");
    asm volatile("s_waitcnt vmcnt(0)");


    asm volatile(
        "v_mov_b32 v16, 0\n\t"
        "v_mov_b32 v17, 0\n\t"
        "v_mov_b32 v18, 0\n\t"
        "v_mov_b32 v19, 0\n\t"
        "v_mov_b32 v20, 0\n\t"
        "v_mov_b32 v21, 0\n\t"
        "v_mov_b32 v22, 0\n\t"
        "v_mov_b32 v23, 0\n\t"
        "v_mov_b32 v24, 0\n\t"
        "v_mov_b32 v25, 0\n\t"
        "v_mov_b32 v26, 0\n\t"
        "v_mov_b32 v27, 0\n\t"
        "v_mov_b32 v28, 0\n\t"
        "v_mov_b32 v29, 0\n\t"
        "v_mov_b32 v30, 0\n\t"
        "v_mov_b32 v31, 0\n\t"
        "v_mov_b32 v32, 0\n\t"
        "v_mov_b32 v33, 0\n\t"
        "v_mov_b32 v34, 0\n\t"
        "v_mov_b32 v35, 0\n\t"
        "v_mov_b32 v36, 0\n\t"
        "v_mov_b32 v37, 0\n\t"
        "v_mov_b32 v38, 0\n\t"
        "v_mov_b32 v39, 0\n\t"
        "v_mov_b32 v40, 0\n\t"
        "v_mov_b32 v41, 0\n\t"
        "v_mov_b32 v42, 0\n\t"
        "v_mov_b32 v43, 0\n\t"
        "v_mov_b32 v44, 0\n\t"
        "v_mov_b32 v45, 0\n\t"
        "v_mov_b32 v46, 0\n\t"
        "v_mov_b32 v47, 0\n\t"
        "v_mov_b32 v48, 0\n\t"
        "v_mov_b32 v49, 0\n\t"
        "v_mov_b32 v50, 0\n\t"
        "v_mov_b32 v51, 0\n\t"
        "v_mov_b32 v52, 0\n\t"
        "v_mov_b32 v53, 0\n\t"
        "v_mov_b32 v54, 0\n\t"
        "v_mov_b32 v55, 0\n\t"
        "v_mov_b32 v56, 0\n\t"
        "v_mov_b32 v57, 0\n\t"
        "v_mov_b32 v58, 0\n\t"
        "v_mov_b32 v59, 0\n\t"
        "v_mov_b32 v60, 0\n\t"
        "v_mov_b32 v61, 0\n\t"
        "v_mov_b32 v62, 0\n\t"
        "v_mov_b32 v63, 0\n\t"
        "v_mov_b32 v64, 0\n\t"
        "v_mov_b32 v65, 0\n\t"
        "v_mov_b32 v66, 0\n\t"
        "v_mov_b32 v67, 0\n\t"
        "v_mov_b32 v68, 0\n\t"
        "v_mov_b32 v69, 0\n\t"
        "v_mov_b32 v70, 0\n\t"
        "v_mov_b32 v71, 0\n\t"
        "v_mov_b32 v72, 0\n\t"
        "v_mov_b32 v73, 0\n\t"
        "v_mov_b32 v74, 0\n\t"
        "v_mov_b32 v75, 0\n\t"
        "v_mov_b32 v76, 0\n\t"
        "v_mov_b32 v77, 0\n\t"
        "v_mov_b32 v78, 0\n\t"
        "v_mov_b32 v79, 0\n\t"
        "v_mov_b32 v80, 0\n\t"
        "v_mov_b32 v81, 0\n\t"
        "v_mov_b32 v82, 0\n\t"
        "v_mov_b32 v83, 0\n\t"
        "v_mov_b32 v84, 0\n\t"
        "v_mov_b32 v85, 0\n\t"
        "v_mov_b32 v86, 0\n\t"
        "v_mov_b32 v87, 0\n\t"
        "v_mov_b32 v88, 0\n\t"
        "v_mov_b32 v89, 0\n\t"
        "v_mov_b32 v90, 0\n\t"
        "v_mov_b32 v91, 0\n\t"
        "v_mov_b32 v92, 0\n\t"
        "v_mov_b32 v93, 0\n\t"
        "v_mov_b32 v94, 0\n\t"
        "v_mov_b32 v95, 0\n\t"
        "v_mov_b32 v96, 0\n\t"
        "v_mov_b32 v97, 0\n\t"
        "v_mov_b32 v98, 0\n\t"
        "v_mov_b32 v99, 0\n\t"
        "v_mov_b32 v100, 0\n\t"
        "v_mov_b32 v101, 0\n\t"
        "v_mov_b32 v102, 0\n\t"
        "v_mov_b32 v103, 0\n\t"
        "v_mov_b32 v104, 0\n\t"
        "v_mov_b32 v105, 0\n\t"
        "v_mov_b32 v106, 0\n\t"
        "v_mov_b32 v107, 0\n\t"
        "v_mov_b32 v108, 0\n\t"
        "v_mov_b32 v109, 0\n\t"
        "v_mov_b32 v110, 0\n\t"
        "v_mov_b32 v111, 0\n\t"
        "v_mov_b32 v112, 0\n\t"
        "v_mov_b32 v113, 0\n\t"
        "v_mov_b32 v114, 0\n\t"
        "v_mov_b32 v115, 0\n\t"
        "v_mov_b32 v116, 0\n\t"
        "v_mov_b32 v117, 0\n\t"
        "v_mov_b32 v118, 0\n\t"
        "v_mov_b32 v119, 0\n\t"
        "v_mov_b32 v120, 0\n\t"
        "v_mov_b32 v121, 0\n\t"
        "v_mov_b32 v122, 0\n\t"
        "v_mov_b32 v123, 0\n\t"
        "v_mov_b32 v124, 0\n\t"
        "v_mov_b32 v125, 0\n\t"
        "v_mov_b32 v126, 0\n\t"
        "v_mov_b32 v127, 0\n\t"
        "v_mov_b32 v128, 0\n\t"
        "v_mov_b32 v129, 0\n\t"
        "v_mov_b32 v130, 0\n\t"
        "v_mov_b32 v131, 0\n\t"
        "v_mov_b32 v132, 0\n\t"
        "v_mov_b32 v133, 0\n\t"
        "v_mov_b32 v134, 0\n\t"
        "v_mov_b32 v135, 0\n\t"
        "v_mov_b32 v136, 0\n\t"
        "v_mov_b32 v137, 0\n\t"
        "v_mov_b32 v138, 0\n\t"
        "v_mov_b32 v139, 0\n\t"
        "v_mov_b32 v140, 0\n\t"
        "v_mov_b32 v141, 0\n\t"
        "v_mov_b32 v142, 0\n\t"
        "v_mov_b32 v143, 0\n\t"
        "v_mov_b32 v144, 0\n\t"
        "v_mov_b32 v145, 0\n\t"
        "v_mov_b32 v146, 0\n\t"
        "v_mov_b32 v147, 0\n\t"
        "v_mov_b32 v148, 0\n\t"
        "v_mov_b32 v149, 0\n\t"
        "v_mov_b32 v150, 0\n\t"
        "v_mov_b32 v151, 0\n\t"
        "v_mov_b32 v152, 0\n\t"
        "v_mov_b32 v153, 0\n\t"
        "v_mov_b32 v154, 0\n\t"
        "v_mov_b32 v155, 0\n\t"
        "v_mov_b32 v156, 0\n\t"
        "v_mov_b32 v157, 0\n\t"
        "v_mov_b32 v158, 0\n\t"
        "v_mov_b32 v159, 0\n\t"
        "v_mov_b32 v160, 0\n\t"
        "v_mov_b32 v161, 0\n\t"
        "v_mov_b32 v162, 0\n\t"
        "v_mov_b32 v163, 0\n\t"
        "v_mov_b32 v164, 0\n\t"
        "v_mov_b32 v165, 0\n\t"
        "v_mov_b32 v166, 0\n\t"
        "v_mov_b32 v167, 0\n\t"
        "v_mov_b32 v168, 0\n\t"
        "v_mov_b32 v169, 0\n\t"
        "v_mov_b32 v170, 0\n\t"
        "v_mov_b32 v171, 0\n\t"
        "v_mov_b32 v172, 0\n\t"
        "v_mov_b32 v173, 0\n\t"
        "v_mov_b32 v174, 0\n\t"
        "v_mov_b32 v175, 0\n\t"
        "v_mov_b32 v176, 0\n\t"
        "v_mov_b32 v177, 0\n\t"
        "v_mov_b32 v178, 0\n\t"
        "v_mov_b32 v179, 0\n\t"
        "v_mov_b32 v180, 0\n\t"
        "v_mov_b32 v181, 0\n\t"
        "v_mov_b32 v182, 0\n\t"
        "v_mov_b32 v183, 0\n\t"
        "v_mov_b32 v184, 0\n\t"
        "v_mov_b32 v185, 0\n\t"
        "v_mov_b32 v186, 0\n\t"
        "v_mov_b32 v187, 0\n\t"
        "v_mov_b32 v188, 0\n\t"
        "v_mov_b32 v189, 0\n\t"
        "v_mov_b32 v190, 0\n\t"
        "v_mov_b32 v191, 0\n\t"
        "v_mov_b32 v192, 0\n\t"
        "v_mov_b32 v193, 0\n\t"
        "v_mov_b32 v194, 0\n\t"
        "v_mov_b32 v195, 0\n\t"
        "v_mov_b32 v196, 0\n\t"
        "v_mov_b32 v197, 0\n\t"
        "v_mov_b32 v198, 0\n\t"
        "v_mov_b32 v199, 0\n\t"
        "v_mov_b32 v200, 0\n\t"
        "v_mov_b32 v201, 0\n\t"
        "v_mov_b32 v202, 0\n\t"
        "v_mov_b32 v203, 0\n\t"
        "v_mov_b32 v204, 0\n\t"
        "v_mov_b32 v205, 0\n\t"
        "v_mov_b32 v206, 0\n\t"
        "v_mov_b32 v207, 0\n\t"
        "v_mov_b32 v208, 0\n\t"
        "v_mov_b32 v209, 0\n\t"
        "v_mov_b32 v210, 0\n\t"
        "v_mov_b32 v211, 0\n\t"
        "v_mov_b32 v212, 0\n\t"
        "v_mov_b32 v213, 0\n\t"
        "v_mov_b32 v214, 0\n\t"
        "v_mov_b32 v215, 0\n\t"
        "v_mov_b32 v216, 0\n\t"
        "v_mov_b32 v217, 0\n\t"
        "v_mov_b32 v218, 0\n\t"
        "v_mov_b32 v219, 0\n\t"
        "v_mov_b32 v220, 0\n\t"
        "v_mov_b32 v221, 0\n\t"
        "v_mov_b32 v222, 0\n\t"
        "v_mov_b32 v223, 0\n\t"
        "v_mov_b32 v224, 0\n\t"
        "v_mov_b32 v225, 0\n\t"
        "v_mov_b32 v226, 0\n\t"
        "v_mov_b32 v227, 0\n\t"
        "v_mov_b32 v228, 0\n\t"
        "v_mov_b32 v229, 0\n\t"
        "v_mov_b32 v230, 0\n\t"
        "v_mov_b32 v231, 0\n\t"
        "v_mov_b32 v232, 0\n\t"
        "v_mov_b32 v233, 0\n\t"
        "v_mov_b32 v234, 0\n\t"
        "v_mov_b32 v235, 0\n\t"
        "v_mov_b32 v236, 0\n\t"
        "v_mov_b32 v237, 0\n\t"
        "v_mov_b32 v238, 0\n\t"
        "v_mov_b32 v239, 0\n\t"
        "v_mov_b32 v240, 0\n\t"
        "v_mov_b32 v241, 0\n\t"
        "v_mov_b32 v242, 0\n\t"
        "v_mov_b32 v243, 0\n\t"
        "v_mov_b32 v244, 0\n\t"
        "v_mov_b32 v245, 0\n\t"
        "v_mov_b32 v246, 0\n\t"
        "v_mov_b32 v247, 0\n\t"
        "v_mov_b32 v248, 0\n\t"
        "v_mov_b32 v249, 0\n\t"
        "v_mov_b32 v250, 0\n\t"
        "v_mov_b32 v251, 0\n\t"
        "v_mov_b32 v252, 0\n\t"
        "v_mov_b32 v253, 0\n\t"
        "v_mov_b32 v254, 0\n\t"
        "v_mov_b32 v255, 0\n\t"
        :
        :
        : "memory"
    );

   
}

// Kernel 2:
__launch_bounds__(256, 1)
__global__ void kernel2_read_vgprs_to_mem_safe(uint32_t *host_buffer, int VGPR_COUNT) {
    const int tid = threadIdx.x;
    uint32_t temp;
    asm volatile(
        "v_mov_b32 %0, 0\n\t"
 	    : "={v255}"(temp)
        :
        : "memory"
    );
    uint32_t tmp;
    int base = tid * VGPR_COUNT;
    #define RD(n, idx) \
        asm volatile("v_mov_b32 %0, v" #n : "=v"(tmp)); \
        host_buffer[base + (idx)] = tmp;

    int idx = 0;
    (void)idx;
    RD(16,  0);  RD(17,  1);  RD(18,  2);  RD(19,  3);
    RD(20,  4);  RD(21,  5);  RD(22,  6);  RD(23,  7);
    RD(24,  8);  RD(25,  9);  RD(26, 10);  RD(27, 11);
    RD(28, 12);  RD(29, 13);  RD(30, 14);  RD(31, 15);
    RD(32, 16);  RD(33, 17);  RD(34, 18);  RD(35, 19);
    RD(36, 20);  RD(37, 21);  RD(38, 22);  RD(39, 23);
    RD(40, 24);  RD(41, 25);  RD(42, 26);  RD(43, 27);
    RD(44, 28);  RD(45, 29);  RD(46, 30);  RD(47, 31);
    RD(48, 32);  RD(49, 33);  RD(50, 34);  RD(51, 35);
    RD(52, 36);  RD(53, 37);  RD(54, 38);  RD(55, 39);
    RD(56, 40);  RD(57, 41);  RD(58, 42);  RD(59, 43);
    RD(60, 44);  RD(61, 45);  RD(62, 46);  RD(63, 47);
    RD(64, 48);  RD(65, 49);  RD(66, 50);  RD(67, 51);
    RD(68, 52);  RD(69, 53);  RD(70, 54);  RD(71, 55);
    RD(72, 56);  RD(73, 57);  RD(74, 58);  RD(75, 59);
    RD(76, 60);  RD(77, 61);  RD(78, 62);  RD(79, 63);
    RD(80, 64);  RD(81, 65);  RD(82, 66);  RD(83, 67);
    RD(84, 68);  RD(85, 69);  RD(86, 70);  RD(87, 71);
    RD(88, 72);  RD(89, 73);  RD(90, 74);  RD(91, 75);
    RD(92, 76);  RD(93, 77);  RD(94, 78);  RD(95, 79);
    RD(96, 80);  RD(97, 81);  RD(98, 82);  RD(99, 83);
    RD(100,84);  RD(101,85);  RD(102,86);  RD(103,87);
    RD(104,88);  RD(105,89);  RD(106,90);  RD(107,91);
    RD(108,92);  RD(109,93);  RD(110,94);  RD(111,95);
    RD(112,96);  RD(113,97);  RD(114,98);  RD(115,99);
    RD(116,100); RD(117,101); RD(118,102); RD(119,103);
    RD(120,104); RD(121,105); RD(122,106); RD(123,107);
    RD(124,108); RD(125,109); RD(126,110); RD(127,111);
    RD(128,112); RD(129,113); RD(130,114); RD(131,115);
    RD(132,116); RD(133,117); RD(134,118); RD(135,119);
    RD(136,120); RD(137,121); RD(138,122); RD(139,123);
    RD(140,124); RD(141,125); RD(142,126); RD(143,127);
    RD(144,128); RD(145,129); RD(146,130); RD(147,131);
    RD(148,132); RD(149,133); RD(150,134); RD(151,135);
    RD(152,136); RD(153,137); RD(154,138); RD(155,139);
    RD(156,140); RD(157,141); RD(158,142); RD(159,143);
    RD(160,144); RD(161,145); RD(162,146); RD(163,147);
    RD(164,148); RD(165,149); RD(166,150); RD(167,151);
    RD(168,152); RD(169,153); RD(170,154); RD(171,155);
    RD(172,156); RD(173,157); RD(174,158); RD(175,159);
    RD(176,160); RD(177,161); RD(178,162); RD(179,163);
    RD(180,164); RD(181,165); RD(182,166); RD(183,167);
    RD(184,168); RD(185,169); RD(186,170); RD(187,171);
    RD(188,172); RD(189,173); RD(190,174); RD(191,175);
    RD(192,176); RD(193,177); RD(194,178); RD(195,179);
    RD(196,180); RD(197,181); RD(198,182); RD(199,183);
    RD(200,184); RD(201,185); RD(202,186); RD(203,187);
    RD(204,188); RD(205,189); RD(206,190); RD(207,191);
    RD(208,192); RD(209,193); RD(210,194); RD(211,195);
    RD(212,196); RD(213,197); RD(214,198); RD(215,199);
    RD(216,200); RD(217,201); RD(218,202); RD(219,203);
    RD(220,204); RD(221,205); RD(222,206); RD(223,207);
    RD(224,208); RD(225,209); RD(226,210); RD(227,211);
    RD(228,212); RD(229,213); RD(230,214); RD(231,215);
    RD(232,216); RD(233,217); RD(234,218); RD(235,219);
    RD(236,220); RD(237,221); RD(238,222); RD(239,223);
    RD(240,224); RD(241,225); RD(242,226); RD(243,227);
    RD(244,228); RD(245,229); RD(246,230); RD(247,231);
    RD(248,232); RD(249,233); RD(250,234); RD(251,235);
    RD(252,236); RD(253,237); RD(254,238); RD(255,239);

    #undef RD
   
    // uint32_t tmp;
    // We'll unroll reads for v16..v255
    //int base = tid * VGPR_COUNT;
    if (tid == 0) {
        asm volatile("v_mov_b32 %0, v16" : "=v"(tmp));
        host_buffer[0] = tmp;
        asm volatile(
		"v_mov_b32 v17, 2 \n\t"
		"s_nop 0 \n\t"
		"v_mov_b32 %0, v17 \n\t"
		"s_waitcnt vmcnt(0) \n\t" : "=v"(tmp));
        host_buffer[1] = tmp;
        

    }
}
