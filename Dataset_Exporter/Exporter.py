import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud


class CSVImageClassificationDatasetExporter(foud.LabeledImageDatasetExporter):
    """Exporter for image classification datasets whose labels and image
    metadata are stored on disk in a CSV file.

    Datasets of this type are exported in the following format:

        <dataset_dir>/
            data/
                <filename1>.<ext>
                <filename2>.<ext>
                ...
            labels.csv

    where ``labels.csv`` is a CSV file in the following format::

        filepath,size_bytes,mime_type,width,height,num_channels,label
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        ...

    Args:
        export_dir: the directory to write the export
    """

    def __init__(self, export_dir):
        super().__init__(export_dir=export_dir)
        self._data_dir = None
        self._labels_path = None
        self._labels = None
        self._image_exporter = None

    @property
    def requires_image_metadata(self):
        """Whether this exporter requires
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each sample
        being exported.
        """
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) exported by this
        exporter.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            exporter directly exports labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the exporter can export a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the exporter can handle label dictionaries with
            value-types specified by this dictionary. Not all keys need be
            present in the exported label dicts
        -   ``None``. In this case, the exporter makes no guarantees about the
            labels that it can export
        """
        return fo.Classification

    def setup(self):
        """Performs any necessary setup before exporting the first sample in
        the dataset.

        This method is called when the exporter's context manager interface is
        entered, :func:`DatasetExporter.__enter__`.
        """
        self._data_dir = os.path.join(self.export_dir, "data")
        self._labels_path = os.path.join(self.export_dir, "labels.csv")
        self._labels = []

        # The `ImageExporter` utility class provides an `export()` method
        # that exports images to an output directory with automatic handling
        # of things like name conflicts
        self._image_exporter = foud.ImageExporter(
            True, export_path=self._data_dir, default_ext=".jpg",
        )
        self._image_exporter.setup()

    def export_sample(self, image_or_path, label, metadata=None):
        """Exports the given sample to the dataset.

        Args:
            image_or_path: an image or the path to the image on disk
            label: an instance of :meth:`label_cls`, or a dictionary mapping
                field names to :class:`fiftyone.core.labels.Label` instances,
                or ``None`` if the sample is unlabeled
            metadata (None): a :class:`fiftyone.core.metadata.ImageMetadata`
                instance for the sample. Only required when
                :meth:`requires_image_metadata` is ``True``
        """
        out_image_path, _ = self._image_exporter.export(image_or_path)

        if metadata is None:
            metadata = fo.ImageMetadata.build_for(image_or_path)

        self._labels.append((
            out_image_path,
            metadata.size_bytes,
            metadata.mime_type,
            metadata.width,
            metadata.height,
            metadata.num_channels,
            label.label,  # here, `label` is a `Classification` instance
        ))

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        exported.

        This method is called when the exporter's context manager interface is
        exited, :func:`DatasetExporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetExporter.__exit__`
        """
        # Ensure the base output directory exists
        basedir = os.path.dirname(self._labels_path)
        if basedir and not os.path.isdir(basedir):
            os.makedirs(basedir)

        # Write the labels CSV file
        with open(self._labels_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filepath",
                "size_bytes",
                "mime_type",
                "width",
                "height",
                "num_channels",
                "label",
            ])
            for row in self._labels:
                writer.writerow(row)
