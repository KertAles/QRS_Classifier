function Classifier( record )
  % Summary of this function and detailed explanation goes here

  % First convert the record into matlab (creates recordm.mat):
  % wfdb2mat -r record
  fileName = sprintf('%s', record);
  fprintf(append('Loading file: ', fileName, '\n'));
  t=cputime();
  [preds, anns] = QRSClassify(fileName);
  fprintf('Running time: %f\n', cputime() - t);
  asciName = sprintf('%s.cls',record);

  fid = fopen(asciName, 'wt');
  for i=1:size(preds,1)
      fprintf(fid,'00:00:00 00/00/0000 %d %s 0 0 0\n', anns(i), preds(i));
  end
  fclose(fid);

  % Now convert the .asc text output to binary WFDB format:
  % wrann -r record -a qrs <record.asc
  % And evaluate against reference annotations (atr) using bxb:
  % bxb -r record -a atr qrs
end

