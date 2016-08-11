#!/bin/bash
echo "The script seems to be working."


ls *.hdr | while read l;
	do
		fsl5.0-fslchfiletype NIFTI_GZ "$l"
		echo "$l" converted to NIFTI
	done

ls *.nii.gz | while read ni;
	do
		fsl5.0-fslmaths "$ni" -bin "$ni"_bin.nii.gz
		echo "$ni" binarized
	done
