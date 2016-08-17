#bash script to go from a mask in MNI space in .hdr format to a binarized NIFTI file in the group template space based on the mni2tmpl matrix

#!/bin/bash
#echo "The script seems to be working."
echo "ROI conversion log" > log.txt

ls *.hdr | while read l;
	do
		fsl5.0-fslchfiletype NIFTI_GZ "$l"
		echo "$l converted to NIFTI at $date" > log.txt
	done

ls *.nii.gz | while read n;
	do
		fsl5.0-fslmaths $n -bin $n
		echo "$n binarized at $date" > log.txt
		fsl5.0-flirt -in $n -ref ~/Documents/JeffOpenFMRI_Task002_Pandora/templates/grpbold7Tp1/from_mni/avg152T1_brain.nii.gz -out $n -init ~/Documents/JeffOpenFMRI_Task002_Pandoras/templates/grpbold7Tp1/xfm/mni2tmpl_12dof.mat -applyxfm
		echo "$n transformed at $date" > log.txt
		fsl5.0-fslmaths $n -bin $n
		echo "$n binarized at $date" > log.txt
	done
